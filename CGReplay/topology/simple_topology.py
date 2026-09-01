#!/usr/bin/env python3
"""
CGReplay Simple Mininet Topology
=================================
1 sender (h1) — 1 switch (s1) — 1 receiver (h2)

IPs match config.yaml:
  h1 (server):  10.0.0.1  server-eth0
  h2 (player):  10.0.0.2  player-eth0

Usage (must run as root):
    sudo python3 simple_topology.py [--protocol rtp|quic] [--bw 10] [--delay 10ms] [--loss 1]

The script:
  1. Builds the topology
  2. Starts tcpdump on h2 (captures QUIC long + short header packets)
  3. Launches server and player in their respective hosts
  4. Copies the PCAP to CGReplay/player/output.pcap when done
"""

import argparse
import os
import sys
import time
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info

# ---------------------------------------------------------------------------
# Defaults (can be overridden by config.yaml or CLI args)
# ---------------------------------------------------------------------------
DEFAULT_BW    = 10     # Mbps
DEFAULT_DELAY = "10ms"
DEFAULT_LOSS  = 0      # %
QUIC_PORT     = 4433
RTP_PORT      = 5002   # used for RTP and SCReAM traffic
PCAP_TMP      = "/tmp/cgquic_capture.pcap"

# Inherit DISPLAY for live watching inside Mininet hosts
_HOST_DISPLAY = os.environ.get("DISPLAY", ":0")

# Resolve paths relative to this script's directory, not the cwd.
# This makes the script work regardless of where sudo is invoked from.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR  = os.path.normpath(os.path.join(_SCRIPT_DIR, "../server"))
PLAYER_DIR  = os.path.normpath(os.path.join(_SCRIPT_DIR, "../player"))
PCAP_DEST   = os.path.join(PLAYER_DIR, "output.pcap")

# When running under sudo, ~ expands to /root instead of the real user's home.
# SUDO_USER contains the original username; fall back to 'hugo' if not set.
_REAL_USER = os.environ.get("SUDO_USER", "hugo")
_USER_HOME = os.path.expanduser(f"~{_REAL_USER}")
VENV     = os.path.join(_USER_HOME, "venv/bin/python3")  # QUIC: aioquic, PyAV, matplotlib
SYS_PY  = "/usr/bin/python3"                              # RTP/SCReAM: gi, GStreamer OpenCV


_CONFIG_PATH = os.path.normpath(os.path.join(_SCRIPT_DIR, "../config/config.yaml"))


def _apply_config(args):
    """Set encoder + frame count in config from CLI; return originals to restore.

    Uses sed so YAML comments survive. Only the `encoding.name` line and the
    `stop_frm_number` line are touched (both are unique in config.yaml).
    """
    import yaml as _yaml
    with open(_CONFIG_PATH) as _f:
        _cfg = _yaml.safe_load(_f)
    orig = {
        "name": _cfg["encoding"]["name"],
        "frames": _cfg["Running"]["stop_frm_number"],
        "width": _cfg["encoding"]["resolution"]["width"],
        "height": _cfg["encoding"]["resolution"]["height"],
        "fps": _cfg["encoding"]["fps"],
        "game": _cfg["Running"]["game"],
    }
    if args.encoder is not None:
        new_name = "H.265" if args.encoder == "h265" else "H.264"
        os.system(f"sed -i 's/^    name: .*/    name: \"{new_name}\"/' {_CONFIG_PATH}")
    if args.frames is not None:
        os.system(f"sed -i 's/^    stop_frm_number: .*/    stop_frm_number: {args.frames}/' {_CONFIG_PATH}")
    if getattr(args, "res", None):
        _w, _h = args.res.lower().replace("×", "x").split("x")
        os.system(f"sed -i 's/^        width: .*/        width: {int(_w)}/' {_CONFIG_PATH}")
        os.system(f"sed -i 's/^        height: .*/        height: {int(_h)}/' {_CONFIG_PATH}")
    if getattr(args, "fps", None):
        os.system(f"sed -i 's/^    fps: .*/    fps: {int(args.fps)}/' {_CONFIG_PATH}")
    if getattr(args, "game", None):
        os.system(f"sed -i 's/^    game: .*/    game: \"{args.game}\"/' {_CONFIG_PATH}")
    return orig


def _restore_config(orig):
    """Restore the config lines changed by _apply_config to their originals."""
    os.system(f"sed -i 's/^    name: .*/    name: \"{orig['name']}\"/' {_CONFIG_PATH}")
    os.system(f"sed -i 's/^    stop_frm_number: .*/    stop_frm_number: {orig['frames']}/' {_CONFIG_PATH}")
    os.system(f"sed -i 's/^        width: .*/        width: {orig['width']}/' {_CONFIG_PATH}")
    os.system(f"sed -i 's/^        height: .*/        height: {orig['height']}/' {_CONFIG_PATH}")
    os.system(f"sed -i 's/^    fps: .*/    fps: {orig['fps']}/' {_CONFIG_PATH}")
    os.system(f"sed -i 's/^    game: .*/    game: \"{orig['game']}\"/' {_CONFIG_PATH}")


def build_topology(bw, bn_bw, delay, jitter, loss):
    """Create and return a Mininet network with 1 server, 1 switch, 1 player.

    Uses OVS in standalone/learning-switch mode (failMode='standalone') so
    no external controller binary is required — works with OVS 2.13+ which
    removed ovs-controller.

    `bw` is the access-link rate (player side); `bn_bw` is the bottleneck rate
    on the server uplink, where the downstream video is shaped. `jitter` is a
    netem string like "5ms" or None for no jitter.
    """

    net = Mininet(
        controller=None,          # no external controller needed
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True,
    )

    info("*** Adding hosts\n")
    h1 = net.addHost("h1", ip="10.0.0.1/24")   # server
    h2 = net.addHost("h2", ip="10.0.0.2/24")   # player

    info("*** Adding switch (standalone learning mode)\n")
    # failMode='standalone' makes OVS act as a self-learning L2 switch
    s1 = net.addSwitch("s1", failMode="standalone")

    info(f"*** Adding links  access={bw}Mbps  bottleneck={bn_bw}Mbps  "
         f"delay={delay}  jitter={jitter}  loss={loss}%\n")
    _h1_kw = dict(bw=bn_bw, delay=delay, loss=loss)   # server uplink = bottleneck
    _h2_kw = dict(bw=bw,    delay=delay, loss=loss)   # player access link
    if jitter:                                        # netem jitter needs a delay set
        _h1_kw["jitter"] = jitter
        _h2_kw["jitter"] = jitter
    net.addLink(h1, s1, cls=TCLink, **_h1_kw)
    net.addLink(h2, s1, cls=TCLink, **_h2_kw)

    return net, h1, h2


def run(args):
    setLogLevel("info")

    # Apply encoder + frame-count from CLI to config (restored in finally below)
    _config_orig = _apply_config(args)
    try:
        _run_inner(args)
    finally:
        _restore_config(_config_orig)


def _run_inner(args):
    # Bottleneck defaults to the access bw when not given; jitter "0"/"0ms"/empty
    # means no netem jitter at all.
    _bn_bw = args.bn_bw if getattr(args, "bn_bw", None) is not None else args.bw
    _jitter = getattr(args, "jitter", None)
    if _jitter and str(_jitter).rstrip("ms").strip() in ("", "0"):
        _jitter = None
    net, h1, h2 = build_topology(args.bw, _bn_bw, args.delay, _jitter, args.loss)

    info("*** Starting network (no external controller needed)\n")
    net.start()

    _enc_name = "H.265" if args.encoder == "h265" else ("H.264" if args.encoder == "h264" else "config")
    info(f"\n{'='*60}\n")
    info(f"  Topology: h1(10.0.0.1) -- s1 -- h2(10.0.0.2)\n")
    info(f"  Protocol: {args.protocol.upper()}   Encoder: {_enc_name}\n")
    info(f"  Access: {args.bw}Mbps  Bottleneck: {_bn_bw}Mbps  {args.delay}  "
         f"jitter={_jitter or '0'}  loss={args.loss}%\n")
    info(f"{'='*60}\n\n")

    # --- PCAP capture (starts before handshake to capture long-header packets) ---
    # Remove any stale PCAP so tcpdump can create the file fresh as root.
    if os.path.exists(PCAP_TMP):
        os.remove(PCAP_TMP)

    # Capture the relevant UDP port for the selected protocol
    capture_port = QUIC_PORT if args.protocol in ("quic", "roq") else RTP_PORT
    info(f"*** Starting PCAP capture on h2 (any iface, UDP port {capture_port})\n")
    info(f"    tcpdump user inside h2: {h2.cmd('id').strip()}\n")
    h2.cmd(
        f"tcpdump -i any -w {PCAP_TMP} udp port {capture_port} "
        f"> /tmp/tcpdump.log 2>&1 & echo $! > /tmp/tcpdump.pid"
    )
    time.sleep(1)  # give tcpdump time to open the capture device
    info(f"    PCAP file after tcpdump start: {h2.cmd(f'ls -la {PCAP_TMP} 2>&1').strip()}\n")
    info(f"    tcpdump PID: {h2.cmd('cat /tmp/tcpdump.pid 2>/dev/null').strip()}\n")

    if args.protocol == "quic":
        _run_quic(h1, h2, args)
    elif args.protocol == "roq":
        _run_roq(h1, h2, args)
    elif args.protocol == "scream":
        _run_scream(h1, h2, args)
    else:
        _run_rtp(h1, h2, args)

    # --- Stop capture: SIGINT → flush + close, then wait for the write to land ---
    info("\n*** Stopping PCAP capture\n")
    h2.cmd("kill -INT $(cat /tmp/tcpdump.pid) 2>/dev/null || true")
    time.sleep(2)  # wait for kernel buffer flush before copying

    os.makedirs(os.path.dirname(PCAP_DEST) or ".", exist_ok=True)
    h2.cmd(f"cp {PCAP_TMP} {PCAP_DEST}")
    info(f"*** PCAP saved to {PCAP_DEST}\n")
    info("    Open with: wireshark output.pcap\n")
    info("    Filter:    quic\n")
    info("    Long header  packets: quic.header_form == 1  (Initial/Handshake)\n")
    info("    Short header packets: quic.header_form == 0  (1-RTT data)\n\n")

    if args.cli:
        info("*** Opening Mininet CLI (type 'exit' to quit)\n")
        CLI(net)

    info("*** Stopping network\n")
    net.stop()

    # Post-processing: compute quality metrics and generate QoE CSV
    if not args.skip_metrics:
        import subprocess as _sp
        _cwd = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))

        # Back up received frames for this mode so later tests don't overwrite them
        _frames_src = os.path.join(PLAYER_DIR, "received_frames")
        _frames_dst = os.path.join(PLAYER_DIR, f"received_frames_{args.protocol}")
        _sp.run(["rm", "-rf", _frames_dst])
        _sp.run(["cp", "-r", _frames_src, _frames_dst])
        info(f"*** Backed up received_frames → received_frames_{args.protocol}/\n")

        # Back up this mode's per-frame + response-time logs. These files are
        # shared across modes and overwritten by each run, so a later test (or a
        # manual post_process re-run) would otherwise read the wrong mode's data.
        _logs = os.path.join(PLAYER_DIR, "logs")
        _frame_log = {"quic": "ply_quic_frame.csv",
                      "roq": "ply_roq_frame.csv"}.get(args.protocol, "ply_frame.csv")
        for _src_name, _dst_name in [
            (_frame_log, f"frame_{args.protocol}.csv"),
            ("responsetime_CG.csv", f"responsetime_{args.protocol}.csv"),
        ]:
            _src = os.path.join(_logs, _src_name)
            if os.path.exists(_src):
                _sp.run(["cp", _src, os.path.join(_logs, _dst_name)])
        info(f"*** Backed up logs → frame_{args.protocol}.csv, responsetime_{args.protocol}.csv\n")

        info(f"\n*** Running post-processing (mode={args.protocol}) ...\n")
        result = _sp.run(
            [VENV, os.path.join(_SCRIPT_DIR, "../tools/post_process.py"),
             "--mode", args.protocol],
            cwd=_cwd,
        )
        if result.returncode == 0:
            info("*** Metrics saved. Run 'python3 tools/plot_qoe.py' to generate figures.\n")
        else:
            info("[WARN] post_process.py returned non-zero — check received_frames/\n")


def _run_quic(h1, h2, args):
    """Launch CGReplay main scripts with QUIC transport."""
    _config = os.path.join(_SCRIPT_DIR, "../config/config.yaml")
    os.system(f"sed -i 's/QUIC: False/QUIC: True/' {_config}")
    os.system(f"sed -i 's/SCReAM: True/SCReAM: False/' {_config}")

    info("*** Launching CGReplay server (QUIC mode) on h1...\n")
    h1.cmd(
        f"cd {SERVER_DIR} && "
        f"DISPLAY={_HOST_DISPLAY} PYTHONUNBUFFERED=1 {VENV} cg_server1.py > /tmp/h1_quic.log 2>&1 &"
    )
    time.sleep(1)  # wait for server to bind

    info("*** Launching CGReplay player (QUIC mode) on h2...\n")
    h2.cmd(
        f"cd {PLAYER_DIR} && "
        f"DISPLAY={_HOST_DISPLAY} PYTHONUNBUFFERED=1 {VENV} cg_gamer1.py > /tmp/h2_quic.log 2>&1 &"
    )

    info("*** Streaming in progress — waiting for completion...\n")
    _wait_for_completion(h2, "/tmp/h2_quic.log", "Receiver finished", timeout=300)

    # Restore config
    _config = os.path.join(_SCRIPT_DIR, "../config/config.yaml")
    os.system(f"sed -i 's/QUIC: True/QUIC: False/' {_config}")

    # Show tail of logs
    info("\n--- h1 server log (tail) ---\n")
    info(h1.cmd("tail -20 /tmp/h1_quic.log"))
    info("\n--- h2 player log (tail) ---\n")
    info(h2.cmd("tail -20 /tmp/h2_quic.log"))


def _run_roq(h1, h2, args):
    """Launch CGReplay with RoQ transport (RTP over QUIC datagrams).

    Same Python pipeline as QUIC (venv: aioquic, PyAV) — only the config flag
    changes, which makes cg_server1/cg_gamer1 delegate to roq_sender/roq_receiver.
    """
    _config = os.path.join(_SCRIPT_DIR, "../config/config.yaml")
    os.system(f"sed -i 's/QUIC: True/QUIC: False/' {_config}")
    os.system(f"sed -i 's/SCReAM: True/SCReAM: False/' {_config}")
    os.system(f"sed -i 's/RoQ: False/RoQ: True/' {_config}")

    info("*** Launching CGReplay server (RoQ mode) on h1...\n")
    h1.cmd(
        f"cd {SERVER_DIR} && "
        f"DISPLAY={_HOST_DISPLAY} PYTHONUNBUFFERED=1 {VENV} cg_server1.py > /tmp/h1_roq.log 2>&1 &"
    )
    time.sleep(1)  # wait for server to bind

    info("*** Launching CGReplay player (RoQ mode) on h2...\n")
    h2.cmd(
        f"cd {PLAYER_DIR} && "
        f"DISPLAY={_HOST_DISPLAY} PYTHONUNBUFFERED=1 {VENV} cg_gamer1.py > /tmp/h2_roq.log 2>&1 &"
    )

    info("*** Streaming in progress — waiting for completion...\n")
    _wait_for_completion(h2, "/tmp/h2_roq.log", "Receiver finished", timeout=300)

    # Restore config (leave all transport flags off — same neutral rest state
    # the QUIC/RTP paths leave behind).
    os.system(f"sed -i 's/RoQ: True/RoQ: False/' {_config}")

    info("\n--- h1 server log (tail) ---\n")
    info(h1.cmd("tail -20 /tmp/h1_roq.log"))
    info("\n--- h2 player log (tail) ---\n")
    info(h2.cmd("tail -20 /tmp/h2_roq.log"))


def _run_rtp(h1, h2, args):
    """Launch original RTP-based CGReplay server and gamer."""
    _config = os.path.join(_SCRIPT_DIR, "../config/config.yaml")
    os.system(f"sed -i 's/QUIC: True/QUIC: False/' {_config}")
    os.system(f"sed -i 's/SCReAM: True/SCReAM: False/' {_config}")

    info("*** Launching RTP server on h1...\n")
    h1.cmd(
        f"cd {SERVER_DIR} && "
        f"DISPLAY={_HOST_DISPLAY} PYTHONUNBUFFERED=1 {SYS_PY} cg_server1.py > /tmp/h1_rtp.log 2>&1 &"
    )
    time.sleep(1)

    info("*** Launching RTP player on h2...\n")
    h2.cmd(
        f"cd {PLAYER_DIR} && "
        f"DISPLAY={_HOST_DISPLAY} PYTHONUNBUFFERED=1 {SYS_PY} cg_gamer1.py > /tmp/h2_rtp.log 2>&1 &"
    )

    info("*** Streaming in progress — waiting for completion...\n")
    _wait_for_completion(h2, "/tmp/h2_rtp.log", "RTP streaming complete", timeout=300)

    info("\n--- h1 server log (tail) ---\n")
    info(h1.cmd("tail -20 /tmp/h1_rtp.log"))
    info("\n--- h2 player log (tail) ---\n")
    info(h2.cmd("tail -20 /tmp/h2_rtp.log"))


def _run_scream(h1, h2, args):
    """Launch CGReplay with SCReAM congestion control."""
    _SCREAM_LIB = os.path.join(_USER_HOME, "CGSynth/scream/code/wrapper_lib")
    _SCREAM_PLUGIN = os.path.join(_USER_HOME, "CGSynth/scream/gstscream/target/debug")

    # Tell sender.sh / receiver.sh which codec to build (h264 default).
    if args.encoder is not None:
        _enc = args.encoder
    else:
        import yaml as _yaml
        with open(_CONFIG_PATH) as _f:
            _enc = ("h265" if _yaml.safe_load(_f)["encoding"]["name"].strip().upper()
                    in ("H.265", "H265", "HEVC") else "h264")

    # Pass the resolution to sender.sh/receiver.sh so the GStreamer caps match
    # the config (videoconvert does NOT rescale — a mismatch deadlocks SCReAM).
    _res_env = ""
    if getattr(args, "res", None):
        _w, _h = args.res.lower().replace("×", "x").split("x")
        _res_env = f"CGREPLAY_WIDTH={int(_w)} CGREPLAY_HEIGHT={int(_h)} "

    _GST_ENV = (
        f"GST_PLUGIN_PATH={_SCREAM_PLUGIN}:${{GST_PLUGIN_PATH:-}} "
        f"LD_LIBRARY_PATH={_SCREAM_LIB}:${{LD_LIBRARY_PATH:-}} "
        f"{_res_env}"
        f"CGREPLAY_ENCODER={_enc}"
    )

    # Temporarily enable SCReAM and disable QUIC in config
    _config = os.path.join(_SCRIPT_DIR, "../config/config.yaml")
    os.system(f"sed -i 's/QUIC: True/QUIC: False/' {_config}")
    os.system(f"sed -i 's/SCReAM: False/SCReAM: True/' {_config}")

    info("*** Launching SCReAM server on h1...\n")
    h1.cmd(
        f"cd {SERVER_DIR} && "
        f"{_GST_ENV} DISPLAY={_HOST_DISPLAY} PYTHONUNBUFFERED=1 {SYS_PY} cg_server1.py > /tmp/h1_scream.log 2>&1 &"
    )
    time.sleep(1)

    info("*** Launching SCReAM player on h2...\n")
    h2.cmd(
        f"cd {PLAYER_DIR} && "
        f"{_GST_ENV} DISPLAY={_HOST_DISPLAY} PYTHONUNBUFFERED=1 {SYS_PY} cg_gamer1.py > /tmp/h2_scream.log 2>&1 &"
    )

    info("*** Streaming in progress — waiting for completion...\n")
    _wait_for_completion(h2, "/tmp/h2_scream.log", "RTP streaming complete", timeout=300)

    # Restore config
    os.system(f"sed -i 's/QUIC: False/QUIC: True/' {_config}")
    os.system(f"sed -i 's/SCReAM: True/SCReAM: False/' {_config}")

    info("\n--- h1 server log (tail) ---\n")
    info(h1.cmd("tail -20 /tmp/h1_scream.log"))
    info("\n--- h2 player log (tail) ---\n")
    info(h2.cmd("tail -20 /tmp/h2_scream.log"))


def _wait_for_completion(host, log_file: str, marker: str, timeout: int = 300):
    """Poll a log file until marker appears or timeout expires."""
    start = time.time()
    while time.time() - start < timeout:
        out = host.cmd(f"grep -c '{marker}' {log_file} 2>/dev/null || echo 0")
        try:
            if int(out.strip()) > 0:
                break
        except ValueError:
            pass
        time.sleep(2)
    else:
        info(f"[WARN] Timeout waiting for '{marker}' in {log_file}\n")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("ERROR: Mininet requires root. Run with: sudo python3 simple_topology.py")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="CGReplay simple Mininet topology: h1 -- s1 -- h2"
    )
    parser.add_argument(
        "--protocol", choices=["rtp", "quic", "roq", "scream"], default="quic",
        help="Transport protocol (default: quic)"
    )
    parser.add_argument(
        "--encoder", choices=["h264", "h265"], default=None,
        help="Video encoder. Sets encoding.name in config (default: leave as-is)"
    )
    parser.add_argument(
        "--frames", type=int, default=None,
        help="Number of frames to stream. Sets Running.stop_frm_number "
             "(default: leave as-is). A run of N frames lasts ~N/fps seconds."
    )
    parser.add_argument(
        "--bw", type=float, default=DEFAULT_BW,
        help=f"Link bandwidth in Mbps (default: {DEFAULT_BW})"
    )
    parser.add_argument(
        "--delay", default=DEFAULT_DELAY,
        help=f"Link delay, e.g. '10ms' (default: {DEFAULT_DELAY})"
    )
    parser.add_argument(
        "--loss", type=float, default=DEFAULT_LOSS,
        help=f"Packet loss percentage (default: {DEFAULT_LOSS})"
    )
    parser.add_argument(
        "--bn-bw", dest="bn_bw", type=float, default=None,
        help="Bottleneck bandwidth in Mbps on the server uplink "
             "(default: same as --bw)"
    )
    parser.add_argument(
        "--jitter", default="0ms",
        help="Link jitter, e.g. '5ms' (default: 0ms = none)"
    )
    parser.add_argument(
        "--res", default=None,
        help="Stream resolution WxH, e.g. '960x540'. Sets encoding.resolution "
             "in config (default: leave as-is)"
    )
    parser.add_argument(
        "--fps", type=int, default=None,
        help="Frames per second. Sets encoding.fps in config (default: leave as-is)"
    )
    parser.add_argument(
        "--game", default=None,
        help="Dataset to stream: Kombat | Forza | Fortnite. Sets Running.game "
             "in config (default: leave as-is)"
    )
    parser.add_argument(
        "--cli", action="store_true",
        help="Drop into Mininet CLI after streaming finishes"
    )
    parser.add_argument(
        "--skip-metrics", action="store_true",
        help="Skip post-processing step (don't run post_process.py)"
    )

    run(parser.parse_args())

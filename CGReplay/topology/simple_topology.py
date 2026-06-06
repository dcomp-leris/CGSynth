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


def build_topology(bw, delay, loss):
    """Create and return a Mininet network with 1 server, 1 switch, 1 player.

    Uses OVS in standalone/learning-switch mode (failMode='standalone') so
    no external controller binary is required — works with OVS 2.13+ which
    removed ovs-controller.
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

    info(f"*** Adding links  bw={bw}Mbps  delay={delay}  loss={loss}%\n")
    net.addLink(h1, s1, cls=TCLink, bw=bw, delay=delay, loss=loss)
    net.addLink(h2, s1, cls=TCLink, bw=bw, delay=delay, loss=loss)

    return net, h1, h2


def run(args):
    setLogLevel("info")

    net, h1, h2 = build_topology(args.bw, args.delay, args.loss)

    info("*** Starting network (no external controller needed)\n")
    net.start()

    info(f"\n{'='*60}\n")
    info(f"  Topology: h1(10.0.0.1) -- s1 -- h2(10.0.0.2)\n")
    info(f"  Protocol: {args.protocol.upper()}\n")
    info(f"  Link: {args.bw}Mbps  {args.delay}  loss={args.loss}%\n")
    info(f"{'='*60}\n\n")

    # --- PCAP capture (starts before handshake to capture long-header packets) ---
    # Remove any stale PCAP so tcpdump can create the file fresh as root.
    if os.path.exists(PCAP_TMP):
        os.remove(PCAP_TMP)

    # Capture the relevant UDP port for the selected protocol
    capture_port = QUIC_PORT if args.protocol == "quic" else RTP_PORT
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
    _GST_ENV = (
        f"GST_PLUGIN_PATH={_SCREAM_PLUGIN}:${{GST_PLUGIN_PATH:-}} "
        f"LD_LIBRARY_PATH={_SCREAM_LIB}:${{LD_LIBRARY_PATH:-}}"
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
        "--protocol", choices=["rtp", "quic", "scream"], default="quic",
        help="Transport protocol (default: quic)"
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
        "--cli", action="store_true",
        help="Drop into Mininet CLI after streaming finishes"
    )
    parser.add_argument(
        "--skip-metrics", action="store_true",
        help="Skip post-processing step (don't run post_process.py)"
    )

    run(parser.parse_args())

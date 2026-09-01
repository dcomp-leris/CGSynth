#!/usr/bin/env python3
"""
CGReplay RoQ Receiver — player side (RTP over QUIC, datagram mode).

Connects to roq_sender.py. Receives RTP packets carried on QUIC datagrams
(RFC 9683 datagram mode), reassembles the fragments of each frame, and decodes.
Frames whose fragments never all arrive are dropped (unreliable) — the point of
RoQ vs plain UDP/RTP is that QUIC's congestion control still applies.

The control channel (Ack / Nack / joystick command) is sent on a reliable QUIC
stream, same as the QUIC receiver.

Run from CGReplay/player/:
    source ~/venv/bin/activate
    python3 roq_receiver.py
"""

import asyncio
import os
import struct
import time
import subprocess
import glob
import yaml
import cv2
import numpy as np
import pandas as pd
import av
from pyzbar import pyzbar
from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.events import (
    StreamDataReceived,
    DatagramFrameReceived,
    HandshakeCompleted,
    ConnectionTerminated,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = "../config/config.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

GAME            = config["Running"]["game"]
STOP_FRAME      = config["Running"]["stop_frm_number"]
SERVER_IP       = config["server"]["server_IP"]
QUIC_PORT       = config["protocols"].get("quic_port", 4433)
PLAYER_IP       = config["gamer"]["player_IP"]
ACK_FREQ        = config["sync"]["ack_freq"]
LIVE_WATCHING   = config["Running"]["live_watching"]
SYNC_FILE       = config[GAME]["sync_file"]
ENC_NAME        = config["encoding"]["name"]                 # "H.264" or "H.265"
DEC_CODEC       = "hevc" if ENC_NAME.strip().upper() in ("H.265", "H265", "HEVC") else "h264"

LOG_DIR = "./logs"
RECV_DIR = "./received_frames"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RECV_DIR, exist_ok=True)

# Clean previous received frames
for f in glob.glob(os.path.join(RECV_DIR, "*.png")):
    os.remove(f)

# RoQ packet headers (must match roq_sender.py)
RTP_FMT   = "!BBHII"
RTP_SIZE  = struct.calcsize(RTP_FMT)    # 12 bytes
FRAG_FMT  = "!IHHd"
FRAG_SIZE = struct.calcsize(FRAG_FMT)   # 16 bytes
HDR_SIZE  = RTP_SIZE + FRAG_SIZE        # 28 bytes

# A frame still incomplete this many frame ids behind the newest one seen is
# considered lost and dropped (small reordering window).
REORDER_WINDOW = 3
IDLE_TIMEOUT   = 5.0    # finish if no datagram arrives for this long (after 1+ frames)

FRAME_LOG = os.path.join(LOG_DIR, "ply_roq_frame.csv")
RATE_LOG  = os.path.join(LOG_DIR, "ratelog_roq.csv")
EVENT_LOG = os.path.join(LOG_DIR, "ply_roq_events.csv")
RT_LOG    = os.path.join(LOG_DIR, "responsetime_CG.csv")

with open(FRAME_LOG, "w") as f:
    f.write("frame_id,size_bytes,recv_time,fps\n")
with open(RATE_LOG, "w") as f:
    f.write("frame_id,fps,cps\n")
with open(EVENT_LOG, "w") as f:
    f.write("timestamp,event,frame_id,size_bytes,fps,cmd_count,response_time_ms\n")
with open(RT_LOG, "w") as f:
    f.write("frame_id,frame_timestamp,cmd_timestamp\n")

# ---------------------------------------------------------------------------
# Sync file loader — same format as cg_gamer1.py / quic_receiver.py
# ---------------------------------------------------------------------------

def load_syncfile(file_path: str) -> pd.DataFrame:
    rows = []
    with open(file_path) as fh:
        next(fh)  # skip header
        for line in fh:
            parts = line.rsplit(",", 1)
            if len(parts) == 2:
                id_cmd, enc = parts
                id_str, cmd_str = id_cmd.split(",", 1)
                rows.append((int(id_str), cmd_str, enc.strip()))
    return pd.DataFrame(rows, columns=["ID", "command", "encrypted_cmd"])

sync_df = load_syncfile(SYNC_FILE)

# ---------------------------------------------------------------------------
# Helpers (identical to quic_receiver.py)
# ---------------------------------------------------------------------------

def decode_h264(data: bytes) -> np.ndarray | None:
    """Decode a single I-frame (H.264 or H.265, independent — no prior context)."""
    try:
        codec = av.CodecContext.create(DEC_CODEC, 'r')
        frames = codec.decode(av.Packet(data))
        if frames:
            return frames[0].to_ndarray(format='bgr24')
    except Exception as e:
        print(f"[WARN] {DEC_CODEC} decode error: {e}")
    return None


def read_qr(frame: np.ndarray) -> tuple[int, str | None]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    codes = pyzbar.decode(blurred)
    for qr in codes:
        data = qr.data.decode()
        for part in data.split(","):
            if "ID:" in part:
                try:
                    return int(part.split(":")[1].strip()), data
                except ValueError:
                    pass
    return -1, None


# ---------------------------------------------------------------------------
# Live display — same separate-process viewer used by the QUIC receiver.
# ---------------------------------------------------------------------------
SYS_PY      = "/usr/bin/python3"
VIEWER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frame_viewer.py")
STOP_FILE   = "/tmp/quic_viewer_stop"


# ---------------------------------------------------------------------------
# RoQ protocol handler
# ---------------------------------------------------------------------------

class RoQReceiverProtocol(QuicConnectionProtocol):
    """
    Reassembles RTP-over-QUIC datagram fragments per frame id, decodes complete
    frames, and drops frames whose fragments never all arrive (unreliable).
    Sends control on a client-initiated reliable stream.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # frame_id -> {frag_index: chunk}, plus expected count and send_time
        self._frags: dict[int, dict[int, bytes]] = {}
        self._frag_count: dict[int, int] = {}
        self._frag_time: dict[int, float] = {}
        self._delivered: set[int] = set()
        self._max_seen = -1
        self._ctrl_stream_id: int | None = None
        self._frame_count = 0
        self._dropped = 0
        self._cmd_count = 0
        self._frame_counter = 1        # tracks sequence for sync file matching
        self._previous_time = time.perf_counter()
        self._cmd_previous_time = time.perf_counter()
        self._current_fps = 0.0
        self._current_cps = 0.0
        self._session_start = time.perf_counter()
        self._last_rx = time.perf_counter()
        self._done = asyncio.Event()
        self._watchdog: asyncio.Task | None = None

    def quic_event_received(self, event):
        if isinstance(event, HandshakeCompleted):
            print("[RoQ] Handshake complete — waiting for RTP datagrams")
            # Open the reliable control stream (client-unidirectional)
            self._ctrl_stream_id = self._quic.get_next_available_stream_id(
                is_unidirectional=True
            )
            self._watchdog = asyncio.ensure_future(self._idle_watchdog())

        elif isinstance(event, DatagramFrameReceived):
            self._last_rx = time.perf_counter()
            self._handle_datagram(event.data)

        elif isinstance(event, ConnectionTerminated):
            print("[RoQ] Connection terminated")
            self._done.set()

    async def _idle_watchdog(self):
        """Finish the session if the stream goes silent (e.g. last frame lost)."""
        while not self._done.is_set():
            await asyncio.sleep(1.0)
            if self._frame_count > 0 and (time.perf_counter() - self._last_rx) > IDLE_TIMEOUT:
                print(f"[RoQ] Idle for {IDLE_TIMEOUT:.0f}s — finishing.")
                self._finish()
                return

    def _handle_datagram(self, data: bytes):
        if len(data) < HDR_SIZE:
            return
        # RTP header is parsed for completeness; frag header carries the app metadata.
        _b0, _b1, _seq, _ts, _ssrc = struct.unpack(RTP_FMT, data[:RTP_SIZE])
        frame_id, frag_index, frag_count, send_time = struct.unpack(
            FRAG_FMT, data[RTP_SIZE:HDR_SIZE])
        chunk = data[HDR_SIZE:]

        if frame_id in self._delivered:
            return  # duplicate / late fragment of an already-decoded frame

        if frame_id > self._max_seen:
            self._max_seen = frame_id

        buf = self._frags.setdefault(frame_id, {})
        buf[frag_index] = chunk
        self._frag_count[frame_id] = frag_count
        self._frag_time.setdefault(frame_id, send_time)

        if len(buf) == frag_count:
            payload = b"".join(buf[i] for i in range(frag_count))
            self._deliver_frame(frame_id, payload, self._frag_time[frame_id])
            self._forget(frame_id, delivered=True)

        # Drop frames that fell too far behind while still incomplete (lost).
        self._purge_stale()

    def _forget(self, frame_id: int, delivered: bool):
        self._frags.pop(frame_id, None)
        self._frag_count.pop(frame_id, None)
        self._frag_time.pop(frame_id, None)
        if delivered:
            self._delivered.add(frame_id)

    def _purge_stale(self):
        cutoff = self._max_seen - REORDER_WINDOW
        stale = [fid for fid in self._frags if fid <= cutoff]
        for fid in stale:
            self._dropped += 1
            have = len(self._frags[fid])
            need = self._frag_count.get(fid, 0)
            print(f"[DROP] frame={fid:04d}  incomplete ({have}/{need} frags)")
            self._forget(fid, delivered=True)  # never deliver it later

    def _deliver_frame(self, frame_id: int, payload: bytes, tx_time: float):
        frame_recv_time = time.perf_counter()
        self._current_fps = 1.0 / max(frame_recv_time - self._previous_time, 1e-6)
        self._previous_time = frame_recv_time

        frame = decode_h264(payload)
        if frame is None:
            print(f"[WARN] frame={frame_id}: {DEC_CODEC} decode failed")
            self._send_control(frame_id, "Nack")
            return

        detected_id, qr_data = read_qr(frame)
        # Atomic write: the live viewer globs *.png and reads the newest file.
        # Writing in place lets it read a half-written PNG (libpng "Read Error").
        # imencode fixes the format from the ".png" arg (not the filename), so
        # the temp name can be one the *.png glob won't match; rename atomically.
        _final = os.path.join(RECV_DIR, f"{frame_id:04d}.png")
        _tmp   = os.path.join(RECV_DIR, f".{frame_id:04d}.png.part")
        _ok, _buf = cv2.imencode(".png", frame)
        if _ok:
            with open(_tmp, "wb") as _fp:
                _fp.write(_buf.tobytes())
            os.replace(_tmp, _final)

        with open(FRAME_LOG, "a") as f:
            f.write(f"{frame_id},{len(payload)},{frame_recv_time:.6f},{self._current_fps:.2f}\n")

        self._frame_count += 1
        n_cmds = 0

        # Ack every ACK_FREQ frames
        if self._frame_count % ACK_FREQ == 0:
            self._send_control(frame_id, "Ack")

        # Send joystick command if sync file has one for this frame position
        matching = sync_df[sync_df["ID"] == self._frame_counter]
        if not matching.empty:
            encrypted_cmds = matching["encrypted_cmd"].values
            n_cmds = len(encrypted_cmds)
            self._send_control(frame_id, "command",
                               cmd=str(encrypted_cmds[0]),
                               number=n_cmds)
            cmd_send_time = time.perf_counter()
            self._cmd_count += n_cmds
            response_time_ms = (cmd_send_time - frame_recv_time) * 1000.0
            print(f"[CMD] frame={frame_id:04d}  commands={n_cmds}  rt={response_time_ms:.1f}ms")
            with open(RT_LOG, "a") as f:
                f.write(f"{frame_id},{frame_recv_time:.6f},{cmd_send_time:.6f}\n")
        else:
            response_time_ms = 0.0

        print(f"[RX]  frame={frame_id:04d}  qr={detected_id:4}  "
              f"size={len(payload):7d}B  fps={self._current_fps:5.1f}")

        with open(EVENT_LOG, "a") as f:
            f.write(f"{frame_recv_time:.6f},FRAME,{frame_id},{len(payload)},{self._current_fps:.2f},{n_cmds},{response_time_ms:.1f}\n")

        self._frame_counter += 1

        if frame_id >= STOP_FRAME - 1:
            self._finish()

    def _finish(self):
        if self._done.is_set():
            return
        duration = time.perf_counter() - self._session_start
        avg_fps = self._frame_count / max(duration, 1e-6)
        print(f"\n[SUMMARY] frames={self._frame_count}  dropped={self._dropped}"
              f"  commands={self._cmd_count}  avg_fps={avg_fps:.1f}  duration={duration:.1f}s")
        cv2.destroyAllWindows()
        self._done.set()

    def _send_control(self, frame_id: int, msg_type: str,
                      cmd: str = "0", number: int = 0):
        if self._ctrl_stream_id is None:
            return
        now = time.perf_counter()
        self._current_cps = 1.0 / max(now - self._cmd_previous_time, 1e-6)
        self._cmd_previous_time = now

        message = (f"{now},{cmd},{frame_id},{msg_type},"
                   f"{number},{self._current_fps:.4f},{self._current_cps:.4f}")
        self._quic.send_stream_data(self._ctrl_stream_id, message.encode())
        self.transmit()

        with open(RATE_LOG, "a") as f:
            f.write(f"{frame_id},{self._current_fps:.4f},{self._current_cps:.4f}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    configuration = QuicConfiguration(is_client=True)
    configuration.verify_mode = False          # self-signed cert in dev
    configuration.alpn_protocols = ["cgroq"]
    configuration.max_datagram_frame_size = 65536   # enable QUIC DATAGRAM extension

    print(f"[RoQ] Connecting to {SERVER_IP}:{QUIC_PORT}")
    print(f"[RoQ] Game={GAME}  stop_frame={STOP_FRAME}  sync={SYNC_FILE}  commands={len(sync_df)}")

    async with connect(
        host=SERVER_IP,
        port=QUIC_PORT,
        configuration=configuration,
        create_protocol=RoQReceiverProtocol,
    ) as protocol:
        await protocol._done.wait()

    print("[RoQ] Receiver finished.")


def run_player():
    """Synchronous entry point (mirrors quic_receiver.run_player)."""
    viewer = None
    if LIVE_WATCHING:
        if os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)
        try:
            viewer = subprocess.Popen([SYS_PY, VIEWER_PATH, os.path.abspath(RECV_DIR)])
            print("[RoQ] Live viewer started (system Python / GTK)")
        except Exception as e:
            print(f"[WARN] could not start live viewer: {e}")

    try:
        asyncio.run(main())
    finally:
        if viewer is not None:
            open(STOP_FILE, "w").close()   # signal the viewer to exit
            try:
                viewer.wait(timeout=3)
            except subprocess.TimeoutExpired:
                viewer.terminate()


if __name__ == "__main__":
    try:
        run_player()
    except KeyboardInterrupt:
        print("\n[RoQ] Receiver stopped.")

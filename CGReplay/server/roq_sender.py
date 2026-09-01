#!/usr/bin/env python3
"""
CGReplay RoQ Sender — server side (RTP over QUIC, datagram mode).

RoQ (RFC 9683) carries RTP packets inside QUIC. This implementation uses the
QUIC DATAGRAM extension (unreliable), the canonical real-time mode: each encoded
frame is packetised into RTP packets and sent as QUIC datagrams. Late or lost
fragments are dropped like real media — QUIC's congestion control still applies,
which is what separates RoQ from plain UDP/RTP (no CC).

Difference from quic_sender.py:
  - QUIC sends one whole frame per reliable stream.
  - RoQ packetises each frame into RTP packets over QUIC datagrams (unreliable).

The player control channel (Ack / Nack / joystick command) stays on a reliable
QUIC stream — control needs reliability.

Run from CGReplay/server/:
    source ~/venv/bin/activate
    python3 roq_sender.py
"""

import asyncio
import os
import struct
import time
import yaml
from fractions import Fraction
import cv2
import numpy as np
import qrcode
import av
from aioquic.asyncio.server import serve
from aioquic.quic.configuration import QuicConfiguration
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.events import (
    StreamDataReceived,
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
FOLDER          = config[GAME]["frames"]
FPS             = config["encoding"]["fps"]
WIDTH           = config["encoding"]["resolution"]["width"]
HEIGHT          = config["encoding"]["resolution"]["height"]
BITRATE_KBPS    = config["encoding"]["starting_bitrate"]
ENC_NAME        = config["encoding"]["name"]                 # "H.264" or "H.265"
IS_H265         = ENC_NAME.strip().upper() in ("H.265", "H265", "HEVC")
QUIC_HOST       = config["server"]["server_IP"]
QUIC_PORT       = config["protocols"].get("quic_port", 4433)
CERT_FILE       = config["protocols"].get("quic_cert", "../../rtp_stream_creation/server.pem")
KEY_FILE        = config["protocols"].get("quic_key",  "../../rtp_stream_creation/server.key")

LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
FRAME_LOG = os.path.join(LOG_DIR, "srv_roq_frame.csv")
EVENT_LOG = os.path.join(LOG_DIR, "srv_roq_events.csv")

# ---------------------------------------------------------------------------
# RoQ / RTP packet format (one datagram = one RTP packet)
#
#   RTP header (12B):  [B version/pad/ext/cc][B marker/pt][H seq][I timestamp][I ssrc]
#   Frag header (16B): [I frame_id][H frag_index][H frag_count][d send_time]
#   payload:           fragment of the encoded frame (<= FRAG_PAYLOAD bytes)
#
# The frag header rides in every datagram so losing the first packet of a frame
# does not lose the frame id / send timestamp needed for logging and RT.
# ---------------------------------------------------------------------------
RTP_FMT      = "!BBHII"
RTP_SIZE     = struct.calcsize(RTP_FMT)     # 12 bytes
FRAG_FMT     = "!IHHd"
FRAG_SIZE    = struct.calcsize(FRAG_FMT)    # 16 bytes
FRAG_PAYLOAD = 1100                         # bytes of frame data per datagram
RTP_PT       = 96                           # dynamic payload type
RTP_SSRC     = 0x43475259                   # "CGRY"
RTP_CLOCK    = 90000                        # RTP video clock (Hz)

with open(FRAME_LOG, "w") as f:
    f.write("frame_id,size_bytes,frags,send_time,fps\n")
with open(EVENT_LOG, "w") as f:
    f.write("timestamp,event,frame_id,size_bytes,fps,ctrl_type,ctrl_count\n")

# ---------------------------------------------------------------------------
# Helpers (identical to quic_sender.py)
# ---------------------------------------------------------------------------

def embed_qr(frame: np.ndarray, frame_id: int, bitrate: int) -> np.ndarray:
    """Overlay a QR code on the bottom-right corner of the frame."""
    qr = qrcode.QRCode(version=1,
                        error_correction=qrcode.constants.ERROR_CORRECT_L,
                        box_size=10, border=4)
    qr.add_data(f"Frame ID: {frame_id}, bitrate:{bitrate}")
    qr.make(fit=True)
    qr_img = np.array(qr.make_image(fill="black", back_color="white").convert("RGB"))
    qr_size = 160
    qr_img = cv2.resize(qr_img, (qr_size, qr_size))
    x = frame.shape[1] - qr_size - 10
    y = frame.shape[0] - qr_size - 10
    frame[y:y + qr_size, x:x + qr_size] = qr_img
    return frame


def create_video_encoder(bitrate_kbps: int) -> av.CodecContext:
    """Create a reusable PyAV encoder (H.264 or H.265, all-I-frame, ultrafast)."""
    name = 'libx265' if IS_H265 else 'libx264'
    codec = av.CodecContext.create(name, 'w')
    codec.width = WIDTH
    codec.height = HEIGHT
    codec.pix_fmt = 'yuv420p'
    codec.bit_rate = bitrate_kbps * 1000
    codec.framerate = Fraction(FPS, 1)
    codec.time_base = Fraction(1, FPS)
    codec.gop_size = 1
    if IS_H265:
        codec.options = {
            'preset': 'ultrafast',
            'tune': 'zerolatency',
            'x265-params': 'repeat-headers=1:keyint=1:min-keyint=1:log-level=none',
        }
    else:
        codec.options = {
            'preset': 'ultrafast',
            'tune': 'zerolatency',
        }
    codec.open()
    return codec


def encode_frame(frame_bgr: np.ndarray, codec: av.CodecContext, pts: int) -> bytes:
    av_frame = av.VideoFrame.from_ndarray(frame_bgr, format='bgr24')
    av_frame.pts = pts
    av_frame.time_base = Fraction(1, FPS)
    return b''.join(bytes(p) for p in codec.encode(av_frame))


def list_frames() -> list[str]:
    files = sorted(f for f in os.listdir(FOLDER) if f.endswith(".png"))
    return [os.path.join(FOLDER, f) for f in files]


# ---------------------------------------------------------------------------
# RoQ protocol handler
# ---------------------------------------------------------------------------

class RoQSenderProtocol(QuicConnectionProtocol):
    """
    One instance per client connection.

    Video frames are packetised into RTP packets carried on QUIC datagrams
    (unreliable). Control messages (Ack / Nack / command) arrive on a
    client-initiated reliable stream.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._streaming_task: asyncio.Task | None = None
        self._bitrate = BITRATE_KBPS
        self._last_ack_frame = 0
        self._frame_count = 0
        self._ctrl_count = 0
        self._seq = 0                       # RTP sequence number (rolls at 16 bits)
        self._session_start = time.perf_counter()
        self._enc = create_video_encoder(self._bitrate)
        print(f"[RoQ] video encoder: {'H.265/libx265' if IS_H265 else 'H.264/libx264'}")

    def quic_event_received(self, event):
        if isinstance(event, HandshakeCompleted):
            print(f"[RoQ] Handshake complete — starting RTP-over-QUIC datagrams (game={GAME})")
            self._streaming_task = asyncio.ensure_future(self._stream_frames())

        elif isinstance(event, StreamDataReceived):
            # Control message from player: "timestamp,cmd,frame_id,type,number,fps,cps"
            try:
                msg = event.data.decode().strip()
                parts = msg.split(",")
                if len(parts) >= 4:
                    msg_type = parts[3]
                    frame_id = int(parts[2])
                    self._last_ack_frame = frame_id
                    self._ctrl_count += 1
                    if msg_type == "command" and len(parts) >= 5:
                        number = parts[4]
                        print(f"[CTRL] frame={frame_id:04d}  type={msg_type}  count={number}")
                        with open(EVENT_LOG, "a") as f:
                            f.write(f"{time.perf_counter():.6f},CTRL,{frame_id},0,0,{msg_type},{number}\n")
                    else:
                        print(f"[CTRL] frame={frame_id:04d}  type={msg_type}")
                        with open(EVENT_LOG, "a") as f:
                            f.write(f"{time.perf_counter():.6f},CTRL,{frame_id},0,0,{msg_type},0\n")
            except Exception:
                pass

        elif isinstance(event, ConnectionTerminated):
            print("[RoQ] Connection terminated")
            if self._streaming_task:
                self._streaming_task.cancel()

    def _send_frame_datagrams(self, frame_id: int, payload: bytes, send_time: float):
        """Packetise one encoded frame into RTP-over-QUIC datagrams."""
        rtp_ts = (frame_id * RTP_CLOCK) // FPS & 0xFFFFFFFF
        n = max(1, (len(payload) + FRAG_PAYLOAD - 1) // FRAG_PAYLOAD)
        for idx in range(n):
            chunk = payload[idx * FRAG_PAYLOAD:(idx + 1) * FRAG_PAYLOAD]
            marker = 1 if idx == n - 1 else 0
            b0 = 0x80                                   # V=2, P=0, X=0, CC=0
            b1 = (marker << 7) | RTP_PT
            rtp = struct.pack(RTP_FMT, b0, b1, self._seq & 0xFFFF, rtp_ts, RTP_SSRC)
            frag = struct.pack(FRAG_FMT, frame_id, idx, n, send_time)
            self._quic.send_datagram_frame(rtp + frag + chunk)
            self._seq += 1
        self.transmit()
        return n

    async def _stream_frames(self):
        frames = list_frames()
        previous_time = time.perf_counter()
        pts = 0

        for path in frames:
            frame_id = int(os.path.basename(path).split(".")[0])
            if frame_id >= STOP_FRAME:
                break

            # Read + preprocess
            frame = cv2.imread(path)
            if frame is None:
                continue
            frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            frame = embed_qr(frame, frame_id, self._bitrate)
            payload = encode_frame(frame, self._enc, pts)
            pts += 1

            send_time = time.perf_counter()
            n_frags = self._send_frame_datagrams(frame_id, payload, send_time)

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now

            with open(FRAME_LOG, "a") as f:
                f.write(f"{frame_id},{len(payload)},{n_frags},{now:.6f},{fps:.2f}\n")
            with open(EVENT_LOG, "a") as f:
                f.write(f"{now:.6f},TX,{frame_id},{len(payload)},{fps:.2f},,\n")

            print(f"[TX]  frame={frame_id:04d}  size={len(payload):7d}B  frags={n_frags:3d}  fps={fps:5.1f}")
            self._frame_count += 1

            # Pace to target FPS
            await asyncio.sleep(1.0 / FPS)

        duration = time.perf_counter() - self._session_start
        avg_fps = self._frame_count / max(duration, 1e-6)
        print(f"\n[SUMMARY] frames_sent={self._frame_count}  ctrl_received={self._ctrl_count}"
              f"  avg_fps={avg_fps:.1f}  duration={duration:.1f}s")
        print("[RoQ] All frames sent.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    configuration = QuicConfiguration(is_client=False)
    configuration.load_cert_chain(CERT_FILE, KEY_FILE)
    configuration.alpn_protocols = ["cgroq"]
    configuration.max_datagram_frame_size = 65536   # enable QUIC DATAGRAM extension

    print(f"[RoQ] Server listening on {QUIC_HOST}:{QUIC_PORT}")
    print(f"[RoQ] Game={GAME}  FPS={FPS}  stop_frame={STOP_FRAME}")

    await serve(
        host=QUIC_HOST,
        port=QUIC_PORT,
        configuration=configuration,
        create_protocol=RoQSenderProtocol,
    )

    await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[RoQ] Server stopped.")

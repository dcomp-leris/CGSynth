# CGReplay++

**Replaying cloud gaming sessions to evaluate transport protocols (UDP, QUIC, RoQ, SCReAM) under controlled network conditions.**

UFSCar LERIS Lab. CGReplay++ extends [CGReplay](https://github.com/dcomp-leris/CGReplay) with multiple transports, a quality-of-experience (QoE) measurement pipeline (VMAF, SSIM, PSNR, response time), and a web dashboard to configure, run, and compare experiments.

CGReplay++ is not plain video streaming. The server streams recorded game frames to the player, and the player sends joystick commands back to the server. This interaction (commands and frames interleaved) is what makes it a faithful cloud gaming workload.

---

## Transports

| Transport | Stack | Congestion control | Status |
|-----------|-------|--------------------|--------|
| UDP / RTP (WebRTC-style) | GStreamer (x264/x265) | none | working |
| QUIC | aioquic + PyAV | none (reliable streams) | working |
| SCReAM v2 | GStreamer + `gstscream` (Ericsson) | SCReAM v2 (L4S) | working |
| RoQ (RTP over QUIC) | aioquic datagrams + PyAV | none (unreliable datagrams) | working |

Codecs: H.264 and H.265 (HEVC), selectable per run.

---

## Repository layout

```
CGReplaypp/
├── server/              # frame server + QUIC sender + game datasets
│   ├── cg_server1.py
│   ├── quic_sender.py
│   └── Kombat/ Forza/ Fortnite/   # recorded frame sequences (large, see Datasets)
├── player/              # player + QUIC receiver + received frames + logs
│   ├── cg_gamer1.py
│   └── quic_receiver.py
├── topology/
│   └── simple_topology.py   # Mininet topology + experiment driver
├── tools/
│   ├── post_process.py      # computes VMAF/SSIM/PSNR/response time/QoE
│   ├── plot_qoe.py          # comparison figures
│   ├── qoe.py               # QoE formula (single source of truth)
│   └── video_Quality.py     # per-frame metric kernels
├── config/
│   └── config.yaml          # encoder, resolution, fps, transport flags
├── ui/                  # CGReplay++ web dashboard (see ui/README.md)
│   ├── backend/         # FastAPI + WebSocket
│   └── frontend/        # React + Vite
└── README.md
```

---

## Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| Linux | Ubuntu 20.04+ | Mininet runs on Linux only |
| Python | 3.9+ | two interpreters used (see below) |
| Mininet | 2.3+ | network emulation, needs root |
| GStreamer | 1.16+ | UDP/RTP and SCReAM paths |
| Node.js | 18+ | web dashboard frontend |
| Rust + CMake | recent | building the SCReAM `gstscream` plugin |

Two Python interpreters are used by design:

- a **virtualenv** for the QUIC path and the tooling (`aioquic`, `PyAV`, `opencv`, `pandas`, `matplotlib`, `streamlit`)
- the **system Python** for the GStreamer (UDP/RTP, SCReAM) paths and for Mininet, which expect the system GStreamer bindings

---

## Install

### 1. Clone

```bash
git clone https://github.com/dcomp-leris/CGReplaypp.git
cd CGReplaypp
```

### 2. Mininet

```bash
sudo apt-get update
sudo apt-get install -y mininet
sudo mn --test pingall   # verify
```

### 3. GStreamer

```bash
sudo apt-get install -y \
  gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly gstreamer1.0-libav \
  python3-gi gir1.2-gstreamer-1.0
```

### 4. Python virtualenv (QUIC + tooling)

```bash
python3 -m venv ~/venv
~/venv/bin/pip install \
  aioquic av opencv-python numpy pandas \
  matplotlib seaborn scikit-image PyYAML qrcode pyzbar streamlit
```

### 5. VMAF (static ffmpeg with libvmaf)

VMAF needs an `ffmpeg` built with `libvmaf`. Download a static build (or compile one) and point `tools/post_process.py` at it:

```bash
ffmpeg -filters | grep libvmaf   # must list libvmaf
```

### 6. SCReAM v2 plugin (for the SCReAM transport)

```bash
git clone https://github.com/EricssonResearch/scream.git
cd scream/gstscream
./scripts/build.sh    # builds the wrapper lib (-DV2) + the Rust plugin
```

This builds **SCReAM v2** (`ScreamV2Tx`), the algorithm in draft `rfc8298bis-screamv2`.

### 7. RoQ

No extra install. RoQ reuses the QUIC virtualenv (`aioquic`, `PyAV`): RTP
packets ride in QUIC DATAGRAM frames (RFC 9683 datagram mode, unreliable),
while the joystick command channel stays on a reliable QUIC stream. See
`server/roq_sender.py` and `player/roq_receiver.py`.

### 8. Web dashboard

```bash
cd ui
bash scripts/setup_linux.sh   # creates backend/.venv and frontend/node_modules
```

---

## Datasets

The recorded frame sequences (`server/Kombat`, `server/Forza`, `server/Fortnite`)
are large (Kombat is ~176 MB) and are not stored in git. Download them separately
and place them under `server/`. Kombat is the default dataset (120 frames, 1280x720).

---

## Run

### Command line (one transport at a time)

```bash
# UDP / RTP
sudo python3 topology/simple_topology.py --protocol rtp    --encoder h264 --bw 10 --delay 10ms --loss 0

# QUIC
sudo python3 topology/simple_topology.py --protocol quic   --encoder h264 --bw 10 --delay 10ms --loss 0

# RoQ (RTP over QUIC datagrams)
sudo python3 topology/simple_topology.py --protocol roq    --encoder h264 --bw 10 --delay 10ms --loss 0

# SCReAM v2
sudo python3 topology/simple_topology.py --protocol scream --encoder h264 --bw 10 --delay 10ms --loss 0
```

Flags: `--protocol {rtp,quic,roq,scream}`, `--encoder {h264,h265}`, `--bw <Mbit>`, `--delay <ms>`, `--loss <%>`, `--frames <N>`.

Each run writes per-frame metrics to `player/logs/metrics_<protocol>.csv`.

### Metrics and figures

```bash
python3 tools/post_process.py --mode quic   # VMAF/SSIM/PSNR/response time/QoE
python3 tools/plot_qoe.py                    # comparison figures across modes
```

### Web dashboard

Demo mode (simulated metrics, no Mininet, any OS):

```bash
cd ui
# backend
./backend/.venv/bin/python backend/server.py
# frontend (second terminal)
cd frontend && npm run dev
# open http://localhost:3000
```

Real mode (measured metrics, Linux + Mininet, root):

```bash
cd ui/backend
sudo USE_REAL_MININET=1 DISPLAY=:0 ./.venv/bin/python server.py
# frontend unchanged; open http://localhost:3000, pick a transport, click Start
```

Open **http://localhost:3000** (the frontend). The backend on :8000 is an API/WebSocket
server and has no home page (a 404 on `/` is expected).

See `ui/README.md` for the dashboard details.

---

## Configuration

`config/config.yaml` controls encoder, resolution, frame rate, bitrate, and the
transport flags. The topology driver and the dashboard write the relevant fields
for each run, so editing it by hand is only needed for advanced cases.

---

## QoE metric

Perceived quality combines video quality and interaction latency:

```
QoE(t) = (VMAF(t) / 100) * exp(-RT(t) / tau)
```

where `RT` is the command response time in milliseconds and `tau` is a latency
sensitivity constant. The formula lives in `tools/qoe.py` as the single source of
truth for both the post-processing pipeline and the dashboard.

---

## Citation

```
Shirmarz, A., de Castro, A. G., Verdi, F. L., & Rothenberg, C. E. (2025).
CGReplay: Capture and Replay of Cloud Gaming Traffic for QoE/QoS Assessment.
arXiv preprint arXiv:2505.11973.
```

# CGReplay

## Setup

### Submodules
This project uses Git submodules. To properly initialize them, run:

```bash
git submodule update --init
```

### Python Environment Setup

#### For Frame Generation Component
The frame generation component requires Python 3.8. You can install it using the deadsnakes PPA:

```bash
# Add deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.8 and required packages
sudo apt install python3.8 python3.8-venv python3.8-dev
```

#### For Quality Metrics Tools
The quality metrics tools require Python 3.12.2. See the Quality Metrics Tools section for setup instructions.

#### OpenCV with GStreamer Support
The CGReplay player component requires OpenCV built with GStreamer support to properly handle video streams. Pre-built packages from pip typically lack this support, so you'll need to build OpenCV from source:

1. First, install the necessary dependencies:
```bash
sudo apt-get install -y build-essential cmake git pkg-config \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev \
    libpng-dev libtiff-dev gfortran openexr \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    python3-dev python3-numpy libatlas-base-dev
```

2. Clone the OpenCV repositories:
```bash
mkdir -p ~/opencv_build && cd ~/opencv_build
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```

3. Configure the build with CMake (adjust paths to match your environment):
```bash
mkdir -p ~/opencv_build/build && cd ~/opencv_build/build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/path/to/your/virtualenv \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D INSTALL_C_EXAMPLES=OFF \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D WITH_GSTREAMER=ON \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
      -D BUILD_EXAMPLES=ON \
      -D PYTHON_EXECUTABLE=/path/to/your/virtualenv/bin/python \
      -D PYTHON_DEFAULT_EXECUTABLE=/path/to/your/virtualenv/bin/python \
      ../opencv
```

4. Build and install (replace `8` with the number of CPU cores you want to use):
```bash
make -j8
make install
```

5. Verify that OpenCV has GStreamer support:
```bash
python -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer
```
   You should see `GStreamer: YES` in the output.

## Project Components

### 1. RTP Video Tools

A pair of Python scripts for working with video streams over RTP networks. These tools enable you to create packet captures (PCAPs) of H.264/H.265 video streams and extract the original video from such captures.

#### Overview

- **rtp_video_packetizer.py**: Converts a series of PNG images into an H.264/H.265 video stream, creates RTP packets for network transmission, and stores them in a PCAP file.
- **rtp_video_extractor.py**: Extracts H.264/H.265 video from RTP packets in a PCAP file and reconstructs the original video stream.

#### Requirements

- Python 3.6+
- Scapy (`pip install scapy`)
- FFmpeg (must be installed and available in your PATH)

Additional Python dependencies:
```bash
pip install scapy
```

#### Usage

##### Creating RTP Video Packets (PCAP)

```bash
python rtp_video_packetizer.py
```

This script will:
1. Read PNG frames from the specified directory
2. Encode them to H.264/H.265 using FFmpeg
3. Packetize the encoded video into RTP packets
4. Save the packets as a PCAP file

Configuration options are defined within the script, including:
- Image source directory
- Output PCAP filename
- Network parameters (IP addresses, ports)
- Video codec (H.264 or H.265)

##### Extracting Video from PCAP

```bash
python rtp_video_extractor.py input_pcap.pcap [-o output_video.mp4] [-c codec]
```

Arguments:
- `input_pcap.pcap`: Path to the input PCAP file (default: 'rtp_stream_h264.pcap')
- `-o, --output`: Path to the output video file (default: 'output.mp4')
- `-c, --codec`: Video codec ('h264' or 'h265', default: 'h264')

Example:
```bash
python rtp_video_extractor.py rtp_stream_h264_fixed.pcap -o extracted_video.mp4 -c h264
```

### 2. Frame Generation and Degradation Toolkit

Located in the `frame_gen` directory, this toolkit provides tools for generating and degrading video frames, as well as evaluating their quality. It includes:

#### Frame Generation
- Frame interpolation and upscaling capabilities
- Support for various video processing algorithms
- Integration with RIFE (Real-time Intermediate Flow Estimation) for high-quality frame interpolation

#### Frame Degradation
- Tools for simulating network conditions
- Frame dropping and quality reduction utilities
- Customizable degradation parameters

#### Quality Assessment
- Comprehensive set of quality metrics tools (see Quality Metrics Tools section)
- Support for both objective and subjective quality evaluation
- Real-time visualization capabilities

For detailed usage instructions, please refer to the documentation in the `frame_gen` directory.

### 3. Quality Metrics Tools

Located in the `frame_gen/tools` directory, these tools help evaluate and analyze video quality metrics, particularly useful for comparing original and processed/interpolated video frames.

#### Setup for Quality Metrics Tools

1. Create a Python virtual environment:
```bash
python3.12.2 -m venv /home/user/venv/cgreplay_metrics
source /home/user/venv/cgreplay_metrics/bin/activate
```

2. Install dependencies:
```bash
pip install -r frame_gen/tools/requirements_tools.txt
```

#### Available Quality Metrics Tools

1. **PSNR and SSIM Analysis** (`frame_gen/tools/psnr_and_ssim.py`)
   - Calculates Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM)
   - Generates plots showing metrics over time
   - Usage: `python frame_gen/tools/psnr_and_ssim.py`

2. **Real-time Quality Metrics Dashboard** (`frame_gen/tools/real_time_quality_metrics.py`)
   - Streamlit-based dashboard for real-time visualization
   - Calculates PSNR, SSIM, LPIPS, and tLPIPS
   - Usage: `streamlit run frame_gen/tools/real_time_quality_metrics.py`

3. **Mean Opinion Score (MOS) Evaluation** (`frame_gen/tools/mean_opinion_score_video_pairs.py`)
   - Tool for subjective quality evaluations (QoE - Quality of Experience)
   - Randomizes video pair order
   - Collects ratings and comments
   - For more details on the QoE subjective evaluation methodology and results, refer to [cgreplay_demo](https://github.com/arielgoes/cgreplay_demo)
   - Usage: `python frame_gen/tools/mean_opinion_score_video_pairs.py`

4. **Video Utilities** (`frame_gen/tools/video_utils.py`)
   - Collection of utility functions for video processing
   - Includes functions for reading, writing, resizing, and frame rate modification
   - Can be imported as a Python module

For detailed usage instructions of each quality metrics tool, please refer to the README.md in the `frame_gen/tools` directory.

## Notes

- All tools require Python 3.12.2 or higher
- Some tools (like LPIPS) require CUDA-capable GPU for optimal performance
- Make sure your input videos/frames have matching dimensions when comparing them
- The tools are designed to work with common video formats (MP4) and image formats (PNG)

## Troubleshooting

- Ensure FFmpeg is properly installed and available in your PATH
- Check that your PNG images are valid and can be encoded by FFmpeg
- For extraction problems, verify that the PCAP contains valid RTP packets with H.264/H.265 payload
- Malformed or incomplete packets in the PCAP file may result in corrupted video output
- For quality metrics tools, ensure you have the correct Python version and all dependencies installed
- For the CGReplay server and player:
  - When a game is specified in `config.yaml` (under `Running:game`), ensure a folder with the same name exists in the `server/` directory (e.g., if game is set to "Kombat", you need a `server/Kombat/` folder)
  - The game folder must contain frames in sequential order
  - Both `player/` and `server/` directories must have a `logs` folder created in them for proper operation
  - The OpenCV in your Python environment must have GStreamer support (see OpenCV with GStreamer Support section)
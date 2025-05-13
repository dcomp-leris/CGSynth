# CGReplay Quality Metrics Tools

This directory contains a set of tools for evaluating and analyzing video quality metrics, particularly useful for comparing original and processed/interpolated video frames.

## Setup

1. Create a Python virtual environment:
```bash
python3.12.2 -m venv /home/user/venv/cgreplay_metrics
source /home/user/venv/cgreplay_metrics/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements_tools.txt
```

## Available Tools

### 1. PSNR and SSIM Analysis (`psnr_and_ssim.py`)

This script calculates Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) between pairs of frames from two different folders. It generates a plot showing the metrics over time.

Usage:
1. Place your original frames in a folder named `original_frames_1920_1080`
2. Place your comparison frames in a folder named `processed_frames_rife_1280_720`
3. Run the script:
```bash
python psnr_and_ssim.py
```

The script will generate a plot showing PSNR and SSIM values for each frame pair, along with average values.

### 2. Real-time Quality Metrics Dashboard (`real_time_quality_metrics.py`)

A Streamlit-based dashboard that provides real-time visualization of quality metrics between two videos. It calculates:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- tLPIPS (Temporal LPIPS)

Usage:
```bash
streamlit run real_time_quality_metrics.py
```

The dashboard allows you to:
- Upload original and processed videos
- View them side by side
- See real-time quality metrics
- Visualize metric trends over time

### 3. Mean Opinion Score (MOS) Evaluation (`mean_opinion_score_video_pairs.py`)

A tool for conducting subjective quality evaluations of video pairs. It:
- Randomizes the order of video pairs
- Collects ratings and comments
- Saves results to a CSV file

Usage:
1. Edit the `video_pairs` list in the script to include your video pairs
2. Run the script:
```bash
python mean_opinion_score_video_pairs.py
```

The script will:
- Generate a unique user ID
- Present video pairs for evaluation
- Collect ratings (1-5) and comments
- Save results to a timestamped CSV file

### 4. Video Utilities (`video_utils.py`)

A collection of utility functions for video processing:

- `read_video()`: Read video frames and FPS
- `write_video()`: Write frames to a video file
- `resize_frames()`: Resize video frames
- `change_fps()`: Modify video frame rate
- `extract_frames()`: Extract frames from a video
- `create_video_from_frames()`: Create a video from a sequence of frames

Usage example:
```python
from video_utils import extract_frames, create_video_from_frames

# Extract frames from a video
frames = extract_frames("input.mp4", "output_frames", frame_rate=30)

# Create a video from frames
create_video_from_frames("output_frames", "output.mp4", fps=30)
```

## Notes

- All tools require Python 3.12.2 or higher
- Some tools (like LPIPS) require CUDA-capable GPU for optimal performance
- Make sure your input videos/frames have matching dimensions when comparing them
- The tools are designed to work with common video formats (MP4) and image formats (PNG) 
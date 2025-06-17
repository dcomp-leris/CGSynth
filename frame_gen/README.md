# Frame Generation and Degradation Toolkit

A comprehensive toolkit for frame generation, degradation simulation, and quality assessment in cloud gaming scenarios.

## Features

- **Frame Degradation Simulation**
  - Network-related artifacts (compression, packet loss, resolution changes)
  - Rendering artifacts (motion blur, color banding)
  - Configurable degradation profiles

- **Frame Interpolation**
  - AI-based methods (RIFE)
  - Traditional methods (simple blending)
  - Support for different resolutions and frame rates

- **Quality Assessment**
  - Objective metrics (PSNR, SSIM, LPIPS)
  - Subjective assessment (MOS)
  - Real-time quality monitoring

## Installation

This project uses different Python versions for different interpolation methods:

- **RIFE**: Python 3.8.20

### Setup Virtual Environments

```bash
# For RIFE (Python 3.8.20)
python3.8 -m venv rife_venv
source rife_venv/bin/activate  # On Linux/Mac
# or
.\rife_venv\Scripts\activate  # On Windows
```

### Install the Package

```bash
# Clone the repository
git clone https://github.com/yourusername/frame_gen.git
cd frame_gen

# Basic installation (no AI methods)
pip install -e .

# Install with specific interpolation methods
pip install -e ".[rife]"    # For RIFE interpolation (Python 3.8.20)
pip install -e ".[metrics]" # For quality metrics

# For development (includes testing and linting tools)
pip install -e ".[dev]"
```

### Requirements by Feature

- **Basic Features** (always installed):
  - numpy
  - opencv-python
  - scipy
  - tqdm
  - pandas
  - matplotlib
  - seaborn

- **RIFE Interpolation** (Python 3.8.20, install with `.[rife]`):
  - torch>=1.7.0
  - torchvision>=0.8.0
  - numpy>=1.19.0
  - opencv-python>=4.5.0
  - tqdm>=4.50.0
  - scipy>=1.7.0
  - lpips>=0.1.4
  - tensorboard>=2.4.0
  - tensorboardX>=2.1
  - einops>=0.3.0
  - pyyaml>=5.4.0
  - matplotlib>=3.3.0
  - pandas>=1.2.0
  - seaborn>=0.11.0

- **Quality Metrics** (install with `.[metrics]`):
  - torch>=1.7.0
  - torchvision>=0.8.0
  - lpips

## Project Structure

```
frame_gen/
├── config/                   # Configuration files
├── data/                     # Data directory
│   ├── original_frames/      # Original video frames
│   │   └── mortal_kombat_11/
│   │       └── 1920_1080/
│   └── processed/            # Processed frames/videos
├── interpolation/            # Interpolation intermediate files 
│   ├── downscaled_original_frames_from_1920_1080_to_1280_720/
│   ├── upscaled_original_frames_from_1280_720_to_1920_1080/
│   ├── models/               # Model weights and configurations
│   └── temp/                 # Temporary files
│       ├── frames/           # Temporary frame storage
│       └── rife/             # RIFE temporary files
├── results/                  # Output results
│   ├── metrics/              # Metric calculation results
│   └── videos/               # Generated videos
├── plots/                    # Visualization outputs
├── src/                      # Source code
│   ├── degradation/          # Frame degradation simulation
│   ├── interpolation/        # Frame interpolation methods
│   ├── metrics/              # Quality assessment metrics
│   ├── evaluation/           # Evaluation code
│   ├── models/               # Model implementations
│   └── utils/                # Utility functions
└── tests/                    # Unit tests
```

## Usage

### Frame Degradation

```python
from frame_gen.src.degradation import CloudGamingNoise

# Initialize the degradation simulator
degrader = CloudGamingNoise(seed=42)

# Create a degradation profile
profile = degrader.create_network_degradation_profile(
    num_frames=100,
    severity=0.5,
    effect_types=['network']
)

# Apply degradation to frames
degraded_frames = degrader.process_frames(original_frames, profile)
```

### Frame Interpolation

```python
# For RIFE interpolation (Python 3.8.20)
from frame_gen.src.interpolation import RIFEInterpolator
rife = RIFEInterpolator()
interpolated_frames = rife.interpolate(frames, target_fps=60)
```

### Quality Assessment

```python
from frame_gen.src.metrics import QualityMetrics

# Initialize metrics
metrics = QualityMetrics()

# Calculate metrics
psnr = metrics.calculate_psnr(original, degraded)
ssim = metrics.calculate_ssim(original, degraded)
lpips = metrics.calculate_lpips(original, degraded)
```

### Video Creation

```python
from frame_gen.src.utils.video_processing import create_video_from_frames

# Create a video from a directory of frames
create_video_from_frames(
    frames_dir="/path/to/frames",
    output_video="/path/to/output.mp4",
    fps=60,
    codec="libx264"
)
```

## Development

```python
# Run all tests
pytest

# Run specific test file
pytest tests/test_interpolation.py

# Run with coverage report
pytest --cov=frame_gen
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

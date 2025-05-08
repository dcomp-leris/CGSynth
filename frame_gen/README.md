# Frame Generation and Degradation Toolkit

A comprehensive toolkit for frame generation, degradation simulation, and quality assessment in cloud gaming scenarios.

## Features

- **Frame Degradation Simulation**
  - Network-related artifacts (compression, packet loss, resolution changes)
  - Rendering artifacts (motion blur, color banding)
  - Configurable degradation profiles

- **Frame Interpolation**
  - AI-based methods (RIFE, FILM)
  - Traditional methods (simple blending)
  - Support for different resolutions and frame rates

- **Quality Assessment**
  - Objective metrics (PSNR, SSIM, LPIPS)
  - Subjective assessment (MOS)
  - Real-time quality monitoring

## Installation

This project uses different Python versions for different interpolation methods:

- **RIFE**: Python 3.8.20
- **FILM**: Python 3.12.2

### Setup Virtual Environments

```bash
# For RIFE (Python 3.8.20)
python3.8 -m venv venv_rife
source venv_rife/bin/activate  # On Linux/Mac
# or
.\venv_rife\Scripts\activate  # On Windows

# For FILM (Python 3.12.2)
python3.12 -m venv venv_film
source venv_film/bin/activate  # On Linux/Mac
# or
.\venv_film\Scripts\activate  # On Windows
```

### Install the Package

```bash
# Clone the repository
git clone https://github.com/yourusername/frame_gen.git
cd frame_gen

# Basic installation (no AI methods)
pip install -e .

# Install with specific interpolation methods
pip install -e ".[film]"    # For FILM interpolation (Python 3.12.2)
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

- **FILM Interpolation** (Python 3.12.2, install with `.[film]`):
  - tensorflow>=2.15.0
  - tensorflow-hub>=0.15.0
  - tensorflow-io-gcs-filesystem>=0.31.0
  - tensorflow-estimator>=2.15.0
  - tensorflow-text>=2.15.0
  - tensorflow-datasets>=4.15.0
  - tensorflow-metadata>=1.14.0
  - tensorflow-probability>=0.23.0
  - tensorflow-recommenders>=0.7.3
  - tensorflow-transform>=1.15.0
  - tensorflow-model-optimization>=0.7.5
  - tensorflow-addons>=0.21.0
  - tensorflow-io>=0.31.0
  - tensorflow-serving-api>=2.15.0
  - tensorflow-gpu>=2.15.0
  - tensorflow-cpu>=2.15.0
  - tensorflow-macos>=2.15.0
  - tensorflow-metal>=1.1.0
  - tensorflow-aarch64>=2.15.0
  - tensorflow-rocm>=2.15.0
  - tensorflow-gpu-configs>=2.15.0
  - tensorflow-gpu-deps>=2.15.0
  - tensorflow-gpu-deps-cuda>=2.15.0
  - tensorflow-gpu-deps-cudnn>=2.15.0
  - tensorflow-gpu-deps-tensorrt>=2.15.0
  - tensorflow-gpu-deps-nccl>=2.15.0
  - tensorflow-gpu-deps-cublas>=2.15.0
  - tensorflow-gpu-deps-cufft>=2.15.0
  - tensorflow-gpu-deps-curand>=2.15.0
  - tensorflow-gpu-deps-cusolver>=2.15.0
  - tensorflow-gpu-deps-cusparse>=2.15.0

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
├── src/
│   ├── degradation/     # Frame degradation simulation
│   ├── interpolation/   # Frame interpolation methods
│   ├── metrics/        # Quality assessment metrics
│   └── utils/          # Utility functions
├── tests/              # Unit tests
├── config/             # Configuration files
├── data/               # Data directory
│   ├── raw/           # Original frames/videos
│   ├── processed/     # Processed frames/videos
│   └── results/       # Analysis results
└── notebooks/         # Jupyter notebooks for analysis
```

## Usage

### Frame Degradation

```python
from frame_gen.degradation import CloudGamingNoise

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
# For FILM interpolation (Python 3.12.2)
from frame_gen.interpolation import FILMInterpolator
film = FILMInterpolator()
interpolated_frames = film.interpolate(frames, target_fps=60)

# For RIFE interpolation (Python 3.8.20)
from frame_gen.interpolation import RIFEInterpolator
rife = RIFEInterpolator()
interpolated_frames = rife.interpolate(frames, target_fps=60)
```

### Quality Assessment

```python
from frame_gen.metrics import QualityMetrics

# Initialize metrics
metrics = QualityMetrics()

# Calculate metrics
psnr = metrics.calculate_psnr(original, degraded)
ssim = metrics.calculate_ssim(original, degraded)
lpips = metrics.calculate_lpips(original, degraded)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
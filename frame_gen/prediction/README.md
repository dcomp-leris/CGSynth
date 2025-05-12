# Frame Prediction

This module uses RIFE (Real-Time Intermediate Flow Estimation) for frame prediction. Given two consecutive frames, it predicts the next frame in the sequence.

## Usage

```bash
python predict_frames.py --game <game_name> --res <resolution> [--generate_video GENERATE_VIDEO]
```

### Arguments

- `--game`: Name of the game (for folder organization)
- `--res`: Resolution in format WIDTHxHEIGHT (e.g., 1920x1080)
- `--generate_video`: Optional flag to generate a video from the predicted frames

### Example

```bash
python predict_frames.py --game mortal_kombat_11 --res 1280x720 --generate_video GENERATE_VIDEO
```

## How it Works

1. Takes two consecutive frames as input
2. Uses RIFE to predict the next frame in the sequence
3. Saves the original frames and predicted frame
4. Optionally generates a video from all frames

## Requirements

- Python 3.8.20
- RIFE submodule (ECCV2022-RIFE)
- OpenCV
- Other dependencies as specified in requirements.txt 
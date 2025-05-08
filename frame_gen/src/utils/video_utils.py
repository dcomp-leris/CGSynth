"""
Utility functions for video processing.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)

def read_video(video_path: Union[str, Path]) -> Tuple[List[np.ndarray], float]:
    """
    Read a video file and return its frames and FPS.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (frames, fps)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    logger.info(f"Read {len(frames)} frames from {video_path}")
    return frames, fps

def write_video(
    frames: List[np.ndarray],
    output_path: Union[str, Path],
    fps: float,
    codec: str = "mp4v"
) -> None:
    """
    Write frames to a video file.
    
    Args:
        frames: List of frames to write
        output_path: Path to save the video
        fps: Frames per second
        codec: Video codec to use
    """
    if not frames:
        raise ValueError("No frames to write")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    logger.info(f"Wrote {len(frames)} frames to {output_path}")

def resize_frames(
    frames: List[np.ndarray],
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR
) -> List[np.ndarray]:
    """
    Resize a list of frames to target size.
    
    Args:
        frames: List of frames to resize
        target_size: Target (width, height)
        interpolation: OpenCV interpolation method
        
    Returns:
        List of resized frames
    """
    return [cv2.resize(frame, target_size, interpolation=interpolation) for frame in frames]

def change_fps(
    frames: List[np.ndarray],
    current_fps: float,
    target_fps: float
) -> List[np.ndarray]:
    """
    Change the frame rate of a video by duplicating or dropping frames.
    
    Args:
        frames: List of frames
        current_fps: Current frame rate
        target_fps: Target frame rate
        
    Returns:
        List of frames at target frame rate
    """
    if current_fps == target_fps:
        return frames
    
    ratio = target_fps / current_fps
    if ratio > 1:
        # Duplicate frames
        new_frames = []
        for frame in frames:
            num_duplicates = int(ratio)
            new_frames.extend([frame] * num_duplicates)
        return new_frames
    else:
        # Drop frames
        indices = np.arange(0, len(frames), 1/ratio)
        return [frames[int(i)] for i in indices]

def extract_frames(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    frame_rate: Optional[float] = None
) -> List[Path]:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save frames
        frame_rate: Optional frame rate to extract (if None, extract all frames)
        
    Returns:
        List of paths to extracted frames
    """
    frames, fps = read_video(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if frame_rate is not None:
        frames = change_fps(frames, fps, frame_rate)
    
    frame_paths = []
    for i, frame in enumerate(frames):
        frame_path = output_dir / f"frame_{i:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)
    
    logger.info(f"Extracted {len(frame_paths)} frames to {output_dir}")
    return frame_paths 
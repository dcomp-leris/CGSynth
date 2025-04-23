import os
import sys
import subprocess
import argparse

def create_video_from_frames(frame_folder, output_path='output_video.mp4', fps=30):
    """
    Uses ffmpeg to create a video from PNG frames.
    """
    print("Creating video with ffmpeg...")

    # Construct the ffmpeg command
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite if output file exists
        '-framerate', str(fps),
        '-i', os.path.join(frame_folder, '%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print("Error during ffmpeg execution:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a video from PNG frames using ffmpeg.")
    parser.add_argument('frame_folder', type=str, help='Path to the folder containing PNG frames (named like 0001.png, 0002.png, ...)')
    parser.add_argument('--output', '-o', type=str, default='output_video.mp4', help='Output video file path')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the video')

    args = parser.parse_args()

    create_video_from_frames(args.frame_folder, args.output, args.fps)

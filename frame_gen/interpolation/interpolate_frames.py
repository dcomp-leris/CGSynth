import os
import cv2
import subprocess
import sys
import argparse
from pathlib import Path
import platform
import venv
import shutil
import site
from importlib import import_module
import socket
import time
import tempfile
import numpy as np
import logging

# Add the RIFE submodule to the Python path
RIFE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ECCV2022-RIFE')
sys.path.append(RIFE_PATH)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Find the repo root (assuming `.git` folder is present at the root)
def get_repo_root():
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    raise FileNotFoundError("Could not find the .git directory to determine repo root.")

def create_video_from_frames(frame_folder, output_path='output_video.mp4', fps=30):
    """
    Generates a video from the frames in the specified folder.
    
    Args:
        frame_folder: Path to the folder containing frames
        output_path: Path where the video will be saved
        fps: Frames per second for the output video
    
    Returns:
        bool: True if video generation was successful, False otherwise
    """
    print(f"Generating video from {frame_folder} at {fps} fps...")
    
    try:
        # Try to import the function from src module
        try:
            src_path = os.path.join(get_repo_root(), "frame_gen", "src")
            if src_path not in sys.path:
                sys.path.append(src_path)
            
            from src.utils.video_utils import create_video_from_frames as src_create_video
            src_create_video(frames_dir=frame_folder, output_video=output_path, fps=fps, codec="libx264")
            print(f"Video successfully saved to {output_path} using src module")
            return True
        except ImportError:
            # Fall back to direct ffmpeg method
            print("Could not import from src module, using direct ffmpeg command...")
            
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
            
            subprocess.run(ffmpeg_cmd, check=True)
            print(f"Video successfully saved to {output_path}")
            return True
    except Exception as e:
        print(f"Error during video generation: {e}")
        return False

def addWeighted_interpolation(frame1, frame3):
    """
    Interpolates between two frames using OpenCV's addWeighted method (simple blending).
    """
    return cv2.addWeighted(frame1, 0.5, frame3, 0.5, 0)

def rife_interpolation(frame1, frame3, model):
    """
    Interpolates between two frames using RIFE (Real-Time Intermediate Flow Estimation).
    This function uses the ECCV2022-RIFE repository's inference_image.py script through a virtual environment.
    """
    print("RIFE interpolation using ECCV2022-RIFE repository...")
    
    # Create temporary directory for frames
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp_frames')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save frames to temporary directory
    frame1_path = os.path.join(temp_dir, 'frame1.png')
    frame3_path = os.path.join(temp_dir, 'frame3.png')
    cv2.imwrite(frame1_path, frame1)
    cv2.imwrite(frame3_path, frame3)
    
    # Get the path to the RIFE repository
    rife_repo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ECCV2022-RIFE')
    rife_venv_path = os.path.join(rife_repo_path, 'rife_venv')
    
    # Create a shell script to run the inference
    script_path = os.path.join(temp_dir, 'run_rife.sh')
    with open(script_path, 'w') as f:
        f.write(f'''#!/bin/bash
source "{rife_venv_path}/bin/activate"
cd "{rife_repo_path}"
python inference_img.py --img "{frame1_path}" "{frame3_path}" --exp 1
''')
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    try:
        # Run the script
        subprocess.run(['bash', script_path], check=True)
        
        # Read the interpolated frame from the output directory
        interpolated_frame_path = os.path.join(rife_repo_path, 'output', 'img1.png')
        interpolated_frame = cv2.imread(interpolated_frame_path)
        
        if interpolated_frame is None:
            raise ValueError("Failed to read interpolated frame from output")
        
        # Clean up temporary files
        shutil.rmtree(temp_dir)
        
        return interpolated_frame
        
    except Exception as e:
        print(f"Error during RIFE interpolation: {e}")
        # Clean up temporary files even if there was an error
        shutil.rmtree(temp_dir)
        raise

def setup_rife_environment():
    """Set up the RIFE environment if it doesn't exist."""
    rife_repo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ECCV2022-RIFE')
    rife_venv_path = os.path.join(rife_repo_path, 'rife_venv')
    
    # Check if RIFE repository exists
    if not os.path.exists(rife_repo_path):
        print("RIFE repository not found. Please ensure the submodule is initialized:")
        print("git submodule update --init")
        return False
    
    # Check if virtual environment exists
    if not os.path.exists(rife_venv_path):
        print("Creating RIFE virtual environment...")
        venv.create(rife_venv_path, with_pip=True)
        
        # Install requirements
        requirements_path = os.path.join(rife_repo_path, 'requirements.txt')
        if os.path.exists(requirements_path):
            subprocess.run([
                os.path.join(rife_venv_path, 'bin', 'pip'),
                'install', '-r', requirements_path
            ], check=True)
        else:
            print("Warning: requirements.txt not found in RIFE repository")
    
    return True

def start_rife_server():
    """Start the RIFE server from the submodule."""
    try:
        # Get the absolute path to the RIFE repository
        current_file = os.path.abspath(__file__)
        rife_repo_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file))), 'ECCV2022-RIFE'))
        
        if not os.path.exists(rife_repo_path):
            raise FileNotFoundError(f"RIFE repository not found at {rife_repo_path}")
            
        sys.path.insert(0, rife_repo_path)  # Add to front of path
        
        # Start the server in a separate process
        server_process = subprocess.Popen(
            [sys.executable, os.path.join(rife_repo_path, 'rife_server.py')],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=rife_repo_path  # Set working directory to RIFE repo
        )
        
        # Wait for server to start
        time.sleep(2)
        return server_process
    except Exception as e:
        logger.error(f"Failed to start RIFE server: {e}")
        return None

def rife_interpolate_client(frame1, frame3, temp_dir):
    """
    Send frames to RIFE server for interpolation.
    Returns the interpolated frame.
    """
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save frames to temp files with absolute paths
    frame1_path = os.path.abspath(os.path.join(temp_dir, 'frame1.png'))
    frame3_path = os.path.abspath(os.path.join(temp_dir, 'frame3.png'))
    output_path = os.path.abspath(os.path.join(temp_dir, 'interpolated.png'))
    
    cv2.imwrite(frame1_path, frame1)
    cv2.imwrite(frame3_path, frame3)
    
    print(f"Saved frames to:")
    print(f"  frame1: {frame1_path}")
    print(f"  frame3: {frame3_path}")
    print(f"  output: {output_path}")
    
    # Try to connect with retries
    max_retries = 3
    retry_delay = 1
    
    for i in range(max_retries):
        try:
            # Connect to server and send request
            with socket.create_connection(('localhost', 50051), timeout=5) as sock:
                msg = f"{frame1_path}|{frame3_path}|{output_path}\n"
                print(f"Sending message to server: {msg.strip()}")
                sock.sendall(msg.encode())
                response = sock.recv(1024).decode()
                
                if not response.startswith("OK"):
                    raise RuntimeError(f"RIFE server error: {response}")
                
                # Read and return interpolated frame
                interpolated_frame = cv2.imread(output_path)
                if interpolated_frame is None:
                    raise RuntimeError(f"Failed to read interpolated frame from {output_path}")
                
                return interpolated_frame
                
        except (ConnectionRefusedError, socket.timeout) as e:
            if i < max_retries - 1:
                print(f"Connection attempt {i+1} failed, retrying...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Failed to connect to RIFE server after {max_retries} attempts: {e}")

def stop_rife_server():
    """Send exit command to RIFE server."""
    with socket.create_connection(('localhost', 50051)) as sock:
        sock.sendall(b"EXIT\n")

def process_frames(method='addWeighted'):
    print(f"Using interpolation method: {method}")

    # Map method names to functions
    interpolation_methods = {
        'addWeighted': addWeighted_interpolation,
        'rife': None   # Will be set if 'rife' is chosen
    }

    # Load models if needed
    rife_server = None
    
    if method == 'rife':
        print("Starting RIFE server...")
        try:
            rife_server = start_rife_server()
            print("RIFE server started successfully")
        except Exception as e:
            print(f"Error starting RIFE server: {e}")
            sys.exit(1)

    # Set the interpolation function
    if method == 'rife':
        rife_temp_dir = os.path.join(os.path.dirname(__file__), 'rife_temp')
        interpolate = lambda frame1, frame2: rife_interpolate_client(frame1, frame2, rife_temp_dir)
    else:
        interpolate = interpolation_methods[method]

    frame_files = sorted([f for f in os.listdir(original_folder) if f.endswith('.png') or f.endswith('.jpg')])

    if len(frame_files) < 2:
        print("Error: Need at least 2 frames in the 'original_frames' folder.")
        return

    # Process frames in pairs, generating interpolated frames
    for i in range(0, len(frame_files) - 2, 2):
        frame1_path = os.path.join(original_folder, frame_files[i])
        frame3_path = os.path.join(original_folder, frame_files[i + 2])

        frame1 = cv2.imread(frame1_path)
        frame3 = cv2.imread(frame3_path)

        # Save original frame
        original_frame_dest = os.path.join(processed_folder, f'{(i+1):04d}.png')
        cv2.imwrite(original_frame_dest, frame1)
        print(f"Saved original frame: {original_frame_dest}")
        
        try:
            # Create interpolated frame
            interpolated_frame = interpolate(frame1, frame3)
            interpolated_frame_dest = os.path.join(processed_folder, f'{(i+2):04d}.png')
            cv2.imwrite(interpolated_frame_dest, interpolated_frame)
            print(f"Saved interpolated frame: {interpolated_frame_dest}")
        except Exception as e:
            print(f"Error during interpolation: {e}")
            sys.exit(1)
    
    # Save the last original frame
    last_frame = cv2.imread(os.path.join(original_folder, frame_files[-1]))
    last_frame_dest = os.path.join(processed_folder, f'{(len(frame_files)-1):04d}.png')
    cv2.imwrite(last_frame_dest, last_frame)
    print(f"Saved last original frame: {last_frame_dest}")

    # Clean up
    if method == 'rife' and rife_server:
        print("Stopping RIFE server...")
        stop_rife_server()
        rife_server.terminate()
        rife_server.wait()

def main():
    parser = argparse.ArgumentParser(description='Interpolate frames using various methods')
    parser.add_argument('--method', type=str, default='addWeighted',
                      choices=['addWeighted', 'rife'],
                      help='Interpolation method to use')
    parser.add_argument('--game', type=str, required=True,
                      help='Game name for folder organization')
    parser.add_argument('--res', type=str, required=True,
                      help='Resolution in format WIDTHxHEIGHT (e.g., 1920x1080)')
    parser.add_argument('--generate_video', type=str, default='GENERATE_VIDEO',
                      help='Whether to generate a video from the frames')
    
    args = parser.parse_args()
    
    # Set up paths
    global original_folder, processed_folder
    original_folder = os.path.join(os.path.dirname(__file__), f'downscaled_original_frames_from_1920_1080_to_{args.res.replace("x", "_")}')
    processed_folder = os.path.join(os.path.dirname(__file__), f'processed_frames_{args.method}_{args.res.replace("x", "_")}')
    
    print(f"normalized_original_folder: ", original_folder)
    print(f"normalized_base_path: ", os.path.dirname(__file__))
    
    # Create output directory if it doesn't exist
    os.makedirs(processed_folder, exist_ok=True)
    
    # Process frames
    process_frames(method=args.method)
    
    # Generate video if requested
    if args.generate_video == 'GENERATE_VIDEO':
        output_video = os.path.join(os.path.dirname(__file__), f'interpolated_{args.method}_{args.res.replace("x", "_")}.mp4')
        create_video_from_frames(processed_folder, output_video)

if __name__ == "__main__":
    main()
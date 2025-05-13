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
import socket
import time
import tempfile
import numpy as np
import logging

# Add the parent directory to the Python path to import from interpolation
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
tools_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools')
if tools_path not in sys.path:
    sys.path.append(tools_path)
from video_utils import create_video_from_frames

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def rife_predict_client(frame1, frame2, temp_dir):
    """
    Send frames to RIFE server for prediction.
    Returns the predicted frame.
    """
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save frames to temp files with absolute paths
    frame1_path = os.path.abspath(os.path.join(temp_dir, 'frame1.png'))
    frame2_path = os.path.abspath(os.path.join(temp_dir, 'frame2.png'))
    output_path = os.path.abspath(os.path.join(temp_dir, 'predicted.png'))
    
    cv2.imwrite(frame1_path, frame1)
    cv2.imwrite(frame2_path, frame2)
    
    print(f"Saved frames to:")
    print(f"  frame1: {frame1_path}")
    print(f"  frame2: {frame2_path}")
    print(f"  output: {output_path}")
    
    # Try to connect with retries
    max_retries = 3
    retry_delay = 1
    
    for i in range(max_retries):
        try:
            # Connect to server and send request
            with socket.create_connection(('localhost', 50051), timeout=5) as sock:
                msg = f"{frame1_path}|{frame2_path}|{output_path}\n"
                print(f"Sending message to server: {msg.strip()}")
                sock.sendall(msg.encode())
                response = sock.recv(1024).decode()
                
                if not response.startswith("OK"):
                    raise RuntimeError(f"RIFE server error: {response}")
                
                # Read and return predicted frame
                predicted_frame = cv2.imread(output_path)
                if predicted_frame is None:
                    raise RuntimeError(f"Failed to read predicted frame from {output_path}")
                
                return predicted_frame
                
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

def process_frames():
    print("Using RIFE for frame prediction")

    # Start RIFE server
    print("Starting RIFE server...")
    try:
        rife_server = start_rife_server()
        print("RIFE server started successfully")
    except Exception as e:
        print(f"Error starting RIFE server: {e}")
        sys.exit(1)

    frame_files = sorted([f for f in os.listdir(original_folder) if f.endswith('.png') or f.endswith('.jpg')])

    if len(frame_files) < 2:
        print("Error: Need at least 2 frames in the 'original_frames' folder.")
        return

    # Process frames in pairs, generating predicted frames
    for i in range(0, len(frame_files) - 1):
        frame1_path = os.path.join(original_folder, frame_files[i])
        frame2_path = os.path.join(original_folder, frame_files[i + 1])

        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)

        # Save original frames
        original_frame1_dest = os.path.join(processed_folder, f'{(i+1):04d}.png')
        original_frame2_dest = os.path.join(processed_folder, f'{(i+2):04d}.png')
        cv2.imwrite(original_frame1_dest, frame1)
        cv2.imwrite(original_frame2_dest, frame2)
        print(f"Saved original frames: {original_frame1_dest}, {original_frame2_dest}")
        
        try:
            # Create predicted frame
            predicted_frame = rife_predict_client(frame1, frame2, os.path.join(os.path.dirname(__file__), 'temp'))
            predicted_frame_dest = os.path.join(processed_folder, f'{(i+3):04d}.png')
            cv2.imwrite(predicted_frame_dest, predicted_frame)
            print(f"Saved predicted frame: {predicted_frame_dest}")
        except Exception as e:
            print(f"Error during prediction: {e}")
            sys.exit(1)

    # Clean up
    if rife_server:
        print("Stopping RIFE server...")
        stop_rife_server()
        rife_server.terminate()
        rife_server.wait()

def main():
    parser = argparse.ArgumentParser(description='Predict frames using RIFE')
    parser.add_argument('--game', type=str, required=True,
                      help='Game name for folder organization')
    parser.add_argument('--res', type=str, required=True,
                      help='Resolution in format WIDTHxHEIGHT (e.g., 1920x1080)')
    parser.add_argument('--generate_video', type=str, default='GENERATE_VIDEO',
                      help='Whether to generate a video from the frames')
    parser.add_argument('--fps', type=int, default=30,
                      help='Frames per second for the output video')
    
    args = parser.parse_args()
    
    # Set up paths
    global original_folder, processed_folder
    original_folder = os.path.join(os.path.dirname(__file__), f'downscaled_original_frames_from_1920_1080_to_{args.res.replace("x", "_")}')
    processed_folder = os.path.join(os.path.dirname(__file__), f'predicted_frames_{args.res.replace("x", "_")}')
    
    print(f"normalized_original_folder: ", original_folder)
    print(f"normalized_base_path: ", os.path.dirname(__file__))
    
    # Create output directory if it doesn't exist
    os.makedirs(processed_folder, exist_ok=True)
    
    # Process frames
    process_frames()
    
    # Generate video if requested
    if args.generate_video == 'GENERATE_VIDEO':
        output_video = os.path.join(os.path.dirname(__file__), f'predicted_{args.res.replace("x", "_")}_{args.fps}.mp4')
        create_video_from_frames(processed_folder, output_video, args.fps)

if __name__ == "__main__":
    main() 
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

def addWeighted_interpolation(frame1, frame3):
    """
    Interpolates between two frames using OpenCV's addWeighted method (simple blending).
    """
    return cv2.addWeighted(frame1, 0.5, frame3, 0.5, 0)

def preprocess_frame(frame):
    """Convert a frame to the format expected by FILM model."""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to float and normalize to [0, 1]
    frame_float = frame_rgb.astype(np.float32) / 255.0
    return frame_float

def postprocess_frame(frame_tensor):
    """Convert model output back to OpenCV format."""
    # Convert from [0, 1] to [0, 255] and back to uint8
    frame_uint8 = (frame_tensor.numpy() * 255.0).astype(np.uint8)
    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
    return frame_bgr

def film_interpolation(frame1, frame3, model):
    """
    Interpolates between two frames using Google's FILM model from TensorFlow Hub.
    """
    print("FILM interpolation (actual model)")
    
    # Preprocess frames
    frame1_processed = preprocess_frame(frame1)
    frame3_processed = preprocess_frame(frame3)
    
    # Create tensors for the model
    frame1_tensor = tf.convert_to_tensor(frame1_processed)
    frame3_tensor = tf.convert_to_tensor(frame3_processed)
    
    # Add batch dimension
    frame1_tensor = tf.expand_dims(frame1_tensor, 0)
    frame3_tensor = tf.expand_dims(frame3_tensor, 0)
    
    # Generate intermediate frame (time = 0.5 for middle frame)
    time_step = tf.constant([[0.5]], dtype=tf.float32)  # Shape needs to be [batch_size, 1]
    
    # Create the input dictionary as expected by the model
    inputs = {
        'x0': frame1_tensor,  # First frame
        'x1': frame3_tensor,  # Second frame
        'time': time_step     # Time step for interpolation
    }
    
    # Run the model with the dictionary input
    # The model returns a dictionary with multiple outputs
    output_dict = model(inputs, training=False)
    
    
    # From the previous output, we know the model returns a dictionary
    if isinstance(output_dict, dict):
        print("Model returned a dictionary. Keys:", output_dict.keys())
        # Extract the 'image' key which contains the interpolated frame
        if 'image' in output_dict:
            interpolated_tensor = output_dict['image']
            # Remove batch dimension if present
            if len(interpolated_tensor.shape) == 4:
                interpolated_tensor = interpolated_tensor[0]
            
            # Convert back to OpenCV format
            interpolated_frame = postprocess_frame(interpolated_tensor)
            return interpolated_frame
        else:
            raise ValueError("Expected 'image' key not found in model output")
    else:
        raise TypeError(f"Unexpected model output type: {type(output_dict)}")

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
    rife_repo_path = os.path.expanduser('~/git/ECCV2022-RIFE')
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
        shutil.rmtree(os.path.join(rife_repo_path, 'output'))
        
        return interpolated_frame
        
    except Exception as e:
        print(f"Error during RIFE interpolation: {e}")
        # Clean up temporary files in case of error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if os.path.exists(os.path.join(rife_repo_path, 'output')):
            shutil.rmtree(os.path.join(rife_repo_path, 'output'))
        raise

def setup_rife_environment():
    """
    Sets up a Python 3.8.20 virtual environment for RIFE with specific dependencies.
    Returns True if setup was successful, False otherwise.
    """
    python_version = platform.python_version_tuple()
    if int(python_version[0]) == 3 and int(python_version[1]) == 8:
        return True  # Current environment is compatible
    
    print("Setting up Python 3.8.20 environment for RIFE...")
    
    # Create a virtual environment
    venv_path = os.path.join(os.path.dirname(__file__), 'rife_venv')
    
    # Remove existing venv if it exists
    if os.path.exists(venv_path):
        print("Removing existing virtual environment...")
        try:
            shutil.rmtree(venv_path)
        except Exception as e:
            print(f"Error removing existing virtual environment: {e}")
            return False
    
    try:
        # First check if python3.8 is available
        try:
            result = subprocess.run(['python3.8', '--version'], check=True, capture_output=True, text=True)
            print(f"Found Python version: {result.stdout.strip()}")
            python_cmd = 'python3.8'
        except subprocess.CalledProcessError:
            print("python3.8 not found. Please install Python 3.8 first.")
            print("On Ubuntu, you can install it with:")
            print("sudo add-apt-repository ppa:deadsnakes/ppa")
            print("sudo apt update")
            print("sudo apt install python3.8 python3.8-venv")
            return False
        
        # Create virtual environment using python3.8
        print(f"Creating virtual environment using {python_cmd}...")
        subprocess.run([python_cmd, '-m', 'venv', venv_path], check=True)
        
        # Verify the virtual environment was created correctly
        if sys.platform == "win32":
            python_path = os.path.join(venv_path, "Scripts", "python.exe")
        else:
            python_path = os.path.join(venv_path, "bin", "python")
        
        if not os.path.exists(python_path):
            print(f"Error: Virtual environment Python executable not found at {python_path}")
            return False
        
        # Verify Python version in the virtual environment
        result = subprocess.run([python_path, '--version'], check=True, capture_output=True, text=True)
        print(f"Virtual environment Python version: {result.stdout.strip()}")
        
    except Exception as e:
        print(f"Error creating virtual environment: {e}")
        return False
    
    # Create a requirements file with all dependencies
    requirements_content = """certifi==2025.4.26
charset-normalizer==3.4.1
decorator==4.4.2
filelock==3.16.1
fsspec==2025.3.0
idna==3.10
imageio==2.35.1
imageio-ffmpeg==0.5.1
Jinja2==3.1.6
MarkupSafe==2.1.5
moviepy==1.0.3
mpmath==1.3.0
networkx==3.1
numpy==1.23.5
opencv-python==4.11.0.86
pillow==10.4.0
proglog==0.1.11
requests==2.32.3
scipy==1.10.1
setuptools==57.5.0
sk-video==1.1.10
sympy==1.13.3
torch==2.4.1
torchvision==0.19.1
tqdm==4.67.1
triton==3.0.0
typing_extensions==4.13.2
urllib3==2.2.3
wheel==0.44.0
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.20.5
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvtx-cu12==12.1.105"""
    
    # Write requirements to a temporary file
    requirements_file = os.path.join(os.path.dirname(__file__), 'rife_requirements.txt')
    with open(requirements_file, 'w') as f:
        f.write(requirements_content)
    
    try:
        print("Upgrading pip...")
        subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip==23.0.1"], check=True)
        
        print("Installing dependencies...")
        subprocess.run([python_path, "-m", "pip", "install", "-r", requirements_file], check=True)
        
        # Clean up the temporary requirements file
        os.remove(requirements_file)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing RIFE dependencies: {e}")
        if os.path.exists(requirements_file):
            os.remove(requirements_file)
        return False

def start_film_server():
    """Start the FILM server."""
    try:
        # Start the server in a separate process
        server_process = subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(__file__), 'film_server.py')],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(2)
        return server_process
    except Exception as e:
        logger.error(f"Failed to start FILM server: {e}")
        return None

def film_interpolate_client(frame1, frame2, temp_dir):
    """
    Send frames to FILM server for interpolation.
    Returns the interpolated frame.
    """
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save frames to temp files
    frame1_path = os.path.join(temp_dir, 'frame1.png')
    frame2_path = os.path.join(temp_dir, 'frame2.png')
    output_path = os.path.join(temp_dir, 'interpolated.png')
    
    cv2.imwrite(frame1_path, frame1)
    cv2.imwrite(frame2_path, frame2)
    
    # Try to connect with retries
    max_retries = 3
    retry_delay = 1
    
    for i in range(max_retries):
        try:
            # Connect to server and send request
            with socket.create_connection(('localhost', 50052), timeout=5) as sock:
                msg = f"{frame1_path}|{frame2_path}|{output_path}\n"
                sock.sendall(msg.encode())
                response = sock.recv(1024).decode()
                
                if not response.startswith("OK"):
                    raise RuntimeError(f"FILM server error: {response}")
                
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
                raise RuntimeError(f"Failed to connect to FILM server after {max_retries} attempts: {e}")

def stop_film_server():
    """Send exit command to FILM server."""
    with socket.create_connection(('localhost', 50052)) as sock:
        sock.sendall(b"EXIT\n")

def process_frames(method='addWeighted'):
    print(f"Using interpolation method: {method}")

    # Map method names to functions
    interpolation_methods = {
        'addWeighted': addWeighted_interpolation,
        'film': None,  # Will be set if 'film' is chosen
        'rife': None   # Will be set if 'rife' is chosen
    }

    # Load models if needed
    film_server = None
    rife_server = None
    
    if method == 'film':
        print("Starting FILM server...")
        try:
            film_server = start_film_server()
            print("FILM server started successfully")
        except Exception as e:
            print(f"Error starting FILM server: {e}")
            sys.exit(1)
    elif method == 'rife':
        print("Starting RIFE server...")
        try:
            rife_server = start_rife_server()
            print("RIFE server started successfully")
        except Exception as e:
            print(f"Error starting RIFE server: {e}")
            sys.exit(1)

    # Set the interpolation function
    if method == 'film':
        film_temp_dir = os.path.join(os.path.dirname(__file__), 'film_temp')
        interpolate = lambda frame1, frame2: film_interpolate_client(frame1, frame2, film_temp_dir)
    elif method == 'rife':
        rife_temp_dir = os.path.join(os.path.dirname(__file__), 'rife_temp')
        interpolate = lambda frame1, frame2: rife_interpolate_client(frame1, frame2, rife_temp_dir)
    else:
        interpolate = interpolation_methods[method]

    frame_files = sorted([f for f in os.listdir(original_folder) if f.endswith('.png') or f.endswith('.jpg')])

    #print("frame_files:", frame_files)
    #sys.exit(0)


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
    if method == 'film' and film_server:
        print("Stopping FILM server...")
        stop_film_server()
        film_server.terminate()
        film_server.wait()
    elif method == 'rife' and rife_server:
        print("Stopping RIFE server...")
        stop_rife_server()
        rife_server.terminate()
        rife_server.wait()

def start_rife_server():
    """Start the RIFE server from the submodule."""
    try:
        # Import the server from the submodule
        from rife_server import main as rife_server_main
        
        # Start the server in a separate process
        server_process = subprocess.Popen(
            [sys.executable, os.path.join(RIFE_PATH, 'rife_server.py')],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
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
    
    # Save frames to temp files
    frame1_path = os.path.join(temp_dir, 'frame1.png')
    frame3_path = os.path.join(temp_dir, 'frame3.png')
    output_path = os.path.join(temp_dir, 'interpolated.png')
    
    cv2.imwrite(frame1_path, frame1)
    cv2.imwrite(frame3_path, frame3)
    
    # Try to connect with retries
    max_retries = 3
    retry_delay = 1
    
    for i in range(max_retries):
        try:
            # Connect to server and send request
            with socket.create_connection(('localhost', 50051), timeout=5) as sock:
                msg = f"{frame1_path}|{frame3_path}|{output_path}\n"
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Frame Interpolation Script')
    parser.add_argument(
        '--method', type=str, required=True,
        choices=['film', 'addWeighted', 'rife'],
        help="Interpolation method to use: 'film', 'addWeighted', or 'rife'"
    )
    parser.add_argument(
        '--game', type=str, required=True,
        choices=['mortal_kombat_11', 'forza_horizon_5', 'fortnite'],
        help="Name of the game, e.g., mortal_kombat_11"
    )
    parser.add_argument(
        '--res', type=str, required=True,
        help="Original resolution, e.g., 1920x1080"
    )

    parser.add_argument(
        '--generate_video', type=bool, default=False,
        help="Generate a video with ffmpeg from the processed frames folder"
    )


    args = parser.parse_args()
    method = args.method
    game = args.game
    res_str = args.res  # Example: "1920x1080"
    res_parts = res_str.lower().split('x')
    generate_video = args.generate_video

    if method == 'film':
        import tensorflow as tf
        import tensorflow_hub as hub
        import numpy as np
    
    if len(res_parts) != 2 or not all(p.isdigit() for p in res_parts):
        print("Resolution format should be like 1920x1080")
        sys.exit(1)
    
    original_res = [int(res_parts[0]), int(res_parts[1])]
    downscaled_res = [1280, 720]
    upscaled_res = [1920, 1080]

    repo_root = get_repo_root() # get root repository of the project
    base_path = os.path.join(repo_root, "frame_gen", "interpolation")
    original_folder = os.path.join(
        base_path, 
        "downscaled_original_frames_from_1920_1080_to_{}_{}".format(
            downscaled_res[0], downscaled_res[1]
        )
    )

    # original_folder = os.path.join(
    #     base_path, 
    #     "downscaled_original_frames_from_{}_{}_to_{}_{}".format(
    #         original_res[0], original_res[1],
    #         downscaled_res[0], downscaled_res[1]
    #     )
    # )

    # we are forcing '1920_1080_to_' prefix here
    original_folder = os.path.join(
        base_path, 
        "downscaled_original_frames_from_1920_1080_to_{}_{}".format(
            downscaled_res[0], downscaled_res[1]
        )
    )

    # Normalize paths for comparison
    normalized_original_folder = os.path.abspath(original_folder)
    normalized_base_path = os.path.abspath(base_path)

    print("normalized_original_folder: ", normalized_original_folder)
    print("normalized_base_path: ", normalized_base_path)

    # Prevent using 1920x1080 if it's under 'original_frames'. NOTE: if you have more than 8GB of VRAM, you can comment this if condition
    if (
        "original_frames" in normalized_original_folder
        and "from_1920_1080" not in normalized_original_folder
        and normalized_original_folder.startswith(normalized_base_path)
    ):
        raise RuntimeError("1920x1080 resolution needs more than 8GB of VRAM. Use a lower resolution.")

    if not os.path.exists(original_folder):
        raise FileNotFoundError(f"Directory '{original_folder}' does not exist. Please check the path or generate the folder.")

    processed_folder = os.path.join(
        base_path, "processed_frames_{}_{}".format(method, res_str.replace('x', '_'))
    )

    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
        os.chmod(processed_folder, 0o777)

    if method == 'film':
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Memory growth setting error: {e}")
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

    process_frames(method=method)


    # Generate videos if the flag is set
    if generate_video:
        print("Generating videos from frames...")
        
        frame_gen_base_path = os.path.join(repo_root, "frame_gen")
        # Path to create_video_from_frames.py script at repo root
        create_video_script = os.path.join(frame_gen_base_path, "create_video_from_frames.py")
        
        # Call the script for original and processed frames using positional argument for frame_folder
        # and --output/--fps for the named arguments
        original_video_cmd = [
            sys.executable,
            create_video_script,
            original_folder,  # First positional argument
            "--output", "original_video.mp4",
            "--fps", "30"
        ]
        
        processed_video_cmd = [
            sys.executable,
            create_video_script,
            processed_folder,  # First positional argument
            "--output", f"interpolated_video_{method}.mp4",
            "--fps", "30"
        ]
        
        try:
            print("Generating original video...")
            subprocess.run(original_video_cmd, check=True)
            
            print("Generating interpolated video...")
            subprocess.run(processed_video_cmd, check=True)
            
            print("Video generation completed.")
        except subprocess.CalledProcessError as e:
            print(f"Error during video generation: {e}")
    else:
        print("Video generation skipped. Use --generate_video flag to create videos.")
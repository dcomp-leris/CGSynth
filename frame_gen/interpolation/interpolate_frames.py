import os
import cv2
import numpy as np
import subprocess
import tensorflow as tf
import tensorflow_hub as hub
import sys
import argparse
from pathlib import Path


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

def process_frames(method='addWeighted'):
    print(f"Using interpolation method: {method}")

    # Map method names to functions
    interpolation_methods = {
        'addWeighted': addWeighted_interpolation,
        'film': None  # Will be set if 'film' is chosen
    }

    # Load FILM model if needed
    film_model = None
    if method == 'film':
        print("Loading FILM model from TensorFlow Hub...")
        try:
            # Load the model from TensorFlow Hub
            film_model = hub.load("https://tfhub.dev/google/film/1")
            print("FILM model loaded successfully")
        except Exception as e:
            print(f"Error loading FILM model: {e}")
            #print("Falling back to addWeighted interpolation")
            #method = 'addWeighted'
            sys.exit(1)

    # Set the interpolation function
    if method == 'film':
        # Create a wrapper function to include the model
        interpolate = lambda frame1, frame3: film_interpolation(frame1, frame3, film_model)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Frame Interpolation Script')
    parser.add_argument(
        '--method', type=str, required=True,
        choices=['film', 'addWeighted'],
        help="Interpolation method to use: 'film' or 'addWeighted'"
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

    #create_video_from_frames(original_folder, output_path='original_video.mp4')
    #create_video_from_frames(processed_folder, output_path='interpolated_video_{}.mp4'.format(method))
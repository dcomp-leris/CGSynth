'''
Copyright 2025 LERIS Lab 

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
''' *********************** Code Updates **************************
Project: CGReplay
Main Module: Capturing
sub Module: Capturing the Screen
Date: 2025-03-10
Author: Alireza Shirmarz
Location: Leris lab
''' ***************************************************************
import time
import numpy as np
import cv2
import mss
import os
import qrcode
import sys
import yaml
from datetime import datetime

# Load configuration from YAML file
with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

FPS = config["fps"]
CAPTURE_DURATION = config["duration"]
FRAMES_DIR = config["frames_dir"]
VIDEO_FILE = f"{config['video_file_prefix']}.mp4"

# Ensure output directory exists
os.makedirs(FRAMES_DIR, exist_ok=True)

def generate_qr_code(data):
    """Generate QR code as an image from the given data."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)

    # Create an image from the QR Code instance
    qr_img = qr.make_image(fill='black', back_color='white')
    
    # Convert to numpy array for OpenCV compatibility
    qr_img = np.array(qr_img.convert('RGB'))
    
    return qr_img

def capture_screen(fps, duration):
    """Capture the full screen and save it as an MP4 video, with frame numbers and QR code overlaid."""
    sct = mss.mss()
    monitor = sct.monitors[1]  # Full screen monitor

    # Calculate frame interval
    frame_interval = 1.0 / fps
    total_frames = int(duration * fps)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 codec
    first_frame = True
    video_writer = None

    print(f"Starting capture. Will run for {duration} seconds...")

    for frame_id in range(total_frames):
        start_time = time.time()

        # Capture screen
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Initialize video writer with frame dimensions
        if first_frame:
            height, width, _ = img.shape
            video_writer = cv2.VideoWriter(VIDEO_FILE, fourcc, fps, (width, height))
            first_frame = False

        # Add the frame number and timestamp text to the frame
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Millisecond accuracy
        frame_text = f"Frame: {frame_id + 1}, Timestamp: {timestamp}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)  # Green color for the text
        thickness = 2
        position = (50, 50)  # Position of the text on the frame (x, y)

        # Add the frame number and timestamp on the top-left corner of the image
        cv2.putText(img, frame_text, position, font, font_scale, font_color, thickness)

        
        # Generate QR code with frame info (frame ID + timestamp)
        qr_data = f"Frame ID: {frame_id + 1}, Timestamp: {timestamp}"
        qr_img = generate_qr_code(qr_data)
        
        # Resize QR code to fit into a corner of the video frame
        qr_size = 100  # QR code size in pixels
        qr_img = cv2.resize(qr_img, (qr_size, qr_size))

        # Overlay the QR code onto the bottom-right corner of the frame
        x_offset = img.shape[1] - qr_size - 10  # 10px padding from the right edge
        y_offset = img.shape[0] - qr_size - 10  # 10px padding from the bottom edge
        img[y_offset:y_offset + qr_size, x_offset:x_offset + qr_size] = qr_img
        
        
        # Write frame to video
        video_writer.write(img)

        # Sleep to maintain the desired frame rate
        elapsed_time = time.time() - start_time
        if elapsed_time < frame_interval:
            time.sleep(frame_interval - elapsed_time)

    # Release the video writer
    video_writer.release()
    print(f"Capture complete. Video saved to {VIDEO_FILE} \n")

if __name__ == "__main__":
    # Start capturing the screen
    capture_screen(fps=FPS, duration=CAPTURE_DURATION)

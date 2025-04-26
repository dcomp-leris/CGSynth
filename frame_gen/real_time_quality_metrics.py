import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import torch
from PIL import Image
import io
import time
import tempfile
import os

# Initialize LPIPS model
loss_fn_alex = lpips.LPIPS(net='alex').cuda()

def calculate_metrics(original, processed):
    """Calculate PSNR, SSIM, and LPIPS between original and processed frames."""
    # Convert to grayscale for SSIM
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    # Calculate PSNR
    psnr_value = psnr(original, processed)
    
    # Calculate SSIM
    ssim_value = ssim(original_gray, processed_gray)
    
    # Calculate LPIPS
    # Convert numpy arrays to PIL Images
    original_pil = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    processed_pil = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    
    # Convert to torch tensors
    original_tensor = lpips.im2tensor(np.array(original_pil))
    processed_tensor = lpips.im2tensor(np.array(processed_pil))
    
    # Move to GPU if available
    if torch.cuda.is_available():
        original_tensor = original_tensor.cuda()
        processed_tensor = processed_tensor.cuda()
    
    lpips_value = loss_fn_alex(original_tensor, processed_tensor).item()
    
    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        'LPIPS': lpips_value
    }

def main():
    st.title("Real-time Quality Metrics Dashboard")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    original_video = st.sidebar.file_uploader("Upload Original Video", type=['mp4'])
    processed_video = st.sidebar.file_uploader("Upload Processed Video", type=['mp4'])
    
    if original_video and processed_video:
        # Create two columns for video display
        col1, col2 = st.columns(2)
        
        # Create placeholders for metrics
        metrics_placeholder = st.empty()
        
        # Save uploaded videos to temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_original:
            tmp_original.write(original_video.read())
            original_path = tmp_original.name
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_processed:
            tmp_processed.write(processed_video.read())
            processed_path = tmp_processed.name
        
        try:
            # Open videos using file paths
            original_cap = cv2.VideoCapture(original_path)
            processed_cap = cv2.VideoCapture(processed_path)
            
            if not original_cap.isOpened() or not processed_cap.isOpened():
                st.error("Error opening video files")
                return
            
            # Create video elements
            original_video_placeholder = col1.empty()
            processed_video_placeholder = col2.empty()
            
            # Create metrics chart
            metrics_chart = st.line_chart()
            
            # Initialize metrics history
            metrics_history = {
                'PSNR': [],
                'SSIM': [],
                'LPIPS': []
            }
            
            # Add a stop button
            stop_button = st.button("Stop")
            
            while True:
                if stop_button:
                    break
                    
                ret1, original_frame = original_cap.read()
                ret2, processed_frame = processed_cap.read()
                
                if not ret1 or not ret2:
                    break
                    
                # Calculate metrics
                metrics = calculate_metrics(original_frame, processed_frame)
                
                # Update metrics history
                for metric, value in metrics.items():
                    metrics_history[metric].append(value)
                
                # Display frames
                original_video_placeholder.image(original_frame, channels="BGR")
                processed_video_placeholder.image(processed_frame, channels="BGR")
                
                # Update metrics display
                metrics_placeholder.write(f"""
                ### Current Metrics
                - PSNR: {metrics['PSNR']:.2f} dB
                - SSIM: {metrics['SSIM']:.3f}
                - LPIPS: {metrics['LPIPS']:.3f}
                """)
                
                # Update metrics chart
                metrics_chart.line_chart(metrics_history)
                
                # Small delay to control playback speed
                time.sleep(0.033)  # ~30fps
                
        finally:
            # Clean up
            original_cap.release()
            processed_cap.release()
            os.unlink(original_path)
            os.unlink(processed_path)

if __name__ == "__main__":
    main() 
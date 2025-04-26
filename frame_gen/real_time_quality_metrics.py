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
import pandas as pd

# Initialize LPIPS model
loss_fn_alex = lpips.LPIPS(net='alex').cuda()

def calculate_tlpips(prev_frame, curr_frame):
    """Calculate temporal LPIPS between consecutive frames."""
    # Convert to PIL Images
    prev_pil = Image.fromarray(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB))
    curr_pil = Image.fromarray(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB))
    
    # Convert to tensors
    prev_tensor = lpips.im2tensor(np.array(prev_pil))
    curr_tensor = lpips.im2tensor(np.array(curr_pil))
    
    if torch.cuda.is_available():
        prev_tensor = prev_tensor.cuda()
        curr_tensor = curr_tensor.cuda()
    
    return loss_fn_alex(prev_tensor, curr_tensor).item()

def calculate_metrics(original, processed, prev_original=None, prev_processed=None):
    """Calculate PSNR, SSIM, LPIPS, and tLPIPS between frames."""
    # Convert to grayscale for SSIM
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    # Calculate PSNR
    psnr_value = psnr(original, processed)
    
    # Calculate SSIM
    ssim_value = ssim(original_gray, processed_gray)
    
    # Calculate LPIPS
    # Convert to PIL Images
    original_pil = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    processed_pil = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    
    original_tensor = lpips.im2tensor(np.array(original_pil))
    processed_tensor = lpips.im2tensor(np.array(processed_pil))
    
    if torch.cuda.is_available():
        original_tensor = original_tensor.cuda()
        processed_tensor = processed_tensor.cuda()
    
    lpips_value = loss_fn_alex(original_tensor, processed_tensor).item()
    
    # Calculate tLPIPS if previous frames are available
    tlpips_value = None
    if prev_original is not None and prev_processed is not None:
        tlpips_value = calculate_tlpips(prev_original, prev_processed)
    
    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        'LPIPS': lpips_value,
        'tLPIPS': tlpips_value
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
            
            # Create metrics charts
            st.subheader("Quality Metrics")
            col1, col2 = st.columns(2)
            
            # Create separate charts for different metric groups
            spatial_chart = col1.empty()  # For PSNR and SSIM
            perceptual_chart = col2.empty()  # For LPIPS and tLPIPS
            
            # Initialize metrics history with time index
            frame_count = 0
            metrics_history = {
                'Spatial': pd.DataFrame(columns=['PSNR', 'SSIM']),
                'Perceptual': pd.DataFrame(columns=['LPIPS', 'tLPIPS'])
            }
            
            # Add a stop button
            stop_button = st.button("Stop")
            
            # Initialize previous frames
            prev_original = None
            prev_processed = None
            
            while True:
                if stop_button:
                    break
                    
                ret1, original_frame = original_cap.read()
                ret2, processed_frame = processed_cap.read()
                
                if not ret1 or not ret2:
                    break
                    
                # Calculate metrics
                metrics = calculate_metrics(original_frame, processed_frame, prev_original, prev_processed)
                
                # Update metrics history with time index
                metrics_history['Spatial'].loc[frame_count] = {
                    'PSNR': metrics['PSNR'],
                    'SSIM': metrics['SSIM']
                }
                
                metrics_history['Perceptual'].loc[frame_count] = {
                    'LPIPS': metrics['LPIPS'],
                    'tLPIPS': metrics['tLPIPS'] if metrics['tLPIPS'] is not None else None
                }
                
                # Display frames
                original_video_placeholder.image(original_frame, channels="BGR")
                processed_video_placeholder.image(processed_frame, channels="BGR")
                
                # Update metrics display
                metrics_text = f"""
                ### Current Metrics
                - PSNR: {metrics['PSNR']:.2f} dB
                - SSIM: {metrics['SSIM']:.3f}
                - LPIPS: {metrics['LPIPS']:.3f}
                """
                
                if metrics['tLPIPS'] is not None:
                    metrics_text += f"- tLPIPS: {metrics['tLPIPS']:.3f}\n"
                
                metrics_placeholder.write(metrics_text)
                
                # Update charts with proper time index
                spatial_chart.line_chart(metrics_history['Spatial'], use_container_width=True)
                perceptual_chart.line_chart(metrics_history['Perceptual'], use_container_width=True)
                
                # Store current frames as previous frames
                prev_original = original_frame.copy()
                prev_processed = processed_frame.copy()
                
                # Increment frame counter
                frame_count += 1
                
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
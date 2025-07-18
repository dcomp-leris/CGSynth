#!/usr/bin/env python3
"""
Video Quality Plots Generator
----------------------------
This script generates PSNR and SSIM plots comparing test videos against a reference video.
It creates separate graphs showing quality metrics over time.

Usage:
    python video_quality_plots.py reference_video.mp4 test_video.mp4 -o output_plot.png
    python video_quality_plots.py reference_video.mp4 test_video1.mp4 test_video2.mp4 -o comparison_plot.png

Author: CGSynth Project
"""

import cv2
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    try:
        return psnr(img1, img2, data_range=255)
    except Exception as e:
        print(f"Error calculating PSNR: {e}")
        return 0.0


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images."""
    try:
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2
        
        return ssim(img1_gray, img2_gray, data_range=255)
    except Exception as e:
        print(f"Error calculating SSIM: {e}")
        return 0.0


def extract_metrics(reference_path, test_path):
    """Extract PSNR and SSIM metrics from two videos."""
    print(f"Analyzing: {os.path.basename(test_path)} vs {os.path.basename(reference_path)}")
    
    # Open video files
    ref_cap = cv2.VideoCapture(reference_path)
    test_cap = cv2.VideoCapture(test_path)
    
    if not ref_cap.isOpened():
        print(f"Error: Could not open reference video {reference_path}")
        return None, None
    
    if not test_cap.isOpened():
        print(f"Error: Could not open test video {test_path}")
        return None, None
    
    # Get video properties
    width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = test_cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
    
    # Initialize metrics storage
    psnr_values = []
    ssim_values = []
    
    # Process frames
    frame_num = 0
    pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")
    
    while True:
        ret_ref, ref_frame = ref_cap.read()
        ret_test, test_frame = test_cap.read()
        
        if not ret_ref or not ret_test:
            break
        
        # Resize frames to match if needed
        if ref_frame.shape != test_frame.shape:
            ref_frame = cv2.resize(ref_frame, (width, height))
            test_frame = cv2.resize(test_frame, (width, height))
        
        # Calculate metrics
        psnr_val = calculate_psnr(ref_frame, test_frame)
        ssim_val = calculate_ssim(ref_frame, test_frame)
        
        # Store metrics
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        
        frame_num += 1
        pbar.update(1)
    
    # Clean up
    pbar.close()
    ref_cap.release()
    test_cap.release()
    
    # Convert to numpy arrays for easier manipulation
    psnr_values = np.array(psnr_values)
    ssim_values = np.array(ssim_values)
    
    # Print summary statistics
    if len(psnr_values) > 0 and len(ssim_values) > 0:
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        min_psnr = np.min(psnr_values)
        max_psnr = np.max(psnr_values)
        min_ssim = np.min(ssim_values)
        max_ssim = np.max(ssim_values)
        
        print(f"Quality Metrics Summary for {os.path.basename(test_path)}:")
        print(f"  Average PSNR: {avg_psnr:.2f} dB")
        print(f"  PSNR Range: {min_psnr:.2f} - {max_psnr:.2f} dB")
        print(f"  Average SSIM: {avg_ssim:.4f}")
        print(f"  SSIM Range: {min_ssim:.4f} - {max_ssim:.4f}")
        print()
    
    return psnr_values, ssim_values


def plot_metrics(reference_path, test_videos, output_path):
    """Generate plots comparing multiple test videos against reference."""
    
    # Extract metrics for each test video
    all_psnr = {}
    all_ssim = {}
    
    for test_video in test_videos:
        video_name = os.path.basename(test_video).replace('.mp4', '')
        psnr_vals, ssim_vals = extract_metrics(reference_path, test_video)
        
        if psnr_vals is not None and ssim_vals is not None:
            all_psnr[video_name] = psnr_vals
            all_ssim[video_name] = ssim_vals
    
    if not all_psnr:
        print("Error: No valid metrics extracted from any video")
        return False
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Color palette for different videos
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot PSNR
    ax1.set_title(f'PSNR Comparison vs {os.path.basename(reference_path)}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('PSNR (dB)')
    ax1.grid(True, alpha=0.3)
    
    for i, (video_name, psnr_vals) in enumerate(all_psnr.items()):
        frames = np.arange(len(psnr_vals))
        color = colors[i % len(colors)]
        ax1.plot(frames, psnr_vals, label=video_name, color=color, linewidth=2)
        
        # Add average line
        avg_psnr = np.mean(psnr_vals)
        ax1.axhline(y=avg_psnr, color=color, linestyle='--', alpha=0.7, 
                   label=f'{video_name} avg: {avg_psnr:.2f} dB')
    
    ax1.legend()
    ax1.set_ylim(bottom=0)
    
    # Plot SSIM
    ax2.set_title(f'SSIM Comparison vs {os.path.basename(reference_path)}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('SSIM')
    ax2.grid(True, alpha=0.3)
    
    for i, (video_name, ssim_vals) in enumerate(all_ssim.items()):
        frames = np.arange(len(ssim_vals))
        color = colors[i % len(colors)]
        ax2.plot(frames, ssim_vals, label=video_name, color=color, linewidth=2)
        
        # Add average line
        avg_ssim = np.mean(ssim_vals)
        ax2.axhline(y=avg_ssim, color=color, linestyle='--', alpha=0.7, 
                   label=f'{video_name} avg: {avg_ssim:.4f}')
    
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Quality metrics plots saved to: {output_path}")
    
    # Also save metrics to CSV
    csv_path = output_path.replace('.png', '_metrics.csv')
    with open(csv_path, 'w') as f:
        f.write("Video,Frame,PSNR,SSIM\n")
        for video_name in all_psnr.keys():
            psnr_vals = all_psnr[video_name]
            ssim_vals = all_ssim[video_name]
            for frame, (psnr_val, ssim_val) in enumerate(zip(psnr_vals, ssim_vals)):
                f.write(f"{video_name},{frame},{psnr_val:.4f},{ssim_val:.6f}\n")
    
    print(f"Metrics data saved to: {csv_path}")
    return True


def main():
    """Main function to parse arguments and generate plots."""
    parser = argparse.ArgumentParser(
        description="Generate PSNR and SSIM quality plots comparing videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_quality_plots.py reference.mp4 test.mp4 -o comparison.png
  python video_quality_plots.py reference.mp4 1Mbit.mp4 10Mbit.mp4 -o bitrate_comparison.png
        """
    )
    
    parser.add_argument("reference_video", help="Path to reference video file")
    parser.add_argument("test_videos", nargs='+', help="Path(s) to test video file(s)")
    parser.add_argument("-o", "--output", required=True, 
                        help="Output plot image file path (PNG)")
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.reference_video):
        print(f"Error: Reference video file not found: {args.reference_video}")
        sys.exit(1)
    
    for test_video in args.test_videos:
        if not os.path.exists(test_video):
            print(f"Error: Test video file not found: {test_video}")
            sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Generating quality plots...")
    print(f"Reference: {args.reference_video}")
    print(f"Test videos: {', '.join(args.test_videos)}")
    print(f"Output: {args.output}")
    print()
    
    # Generate plots
    success = plot_metrics(args.reference_video, args.test_videos, args.output)
    
    if success:
        print("\nPlot generation completed successfully!")
        print(f"Open {args.output} to view the quality comparison plots.")
    else:
        print("\nError: Plot generation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

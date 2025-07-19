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
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse
from skimage import filters
from scipy import ndimage
from scipy.stats import entropy
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


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


def calculate_mse(img1, img2):
    """Calculate Mean Squared Error between two images."""
    try:
        return mse(img1, img2)
    except Exception as e:
        print(f"Error calculating MSE: {e}")
        return 0.0


def calculate_rmse(img1, img2):
    """Calculate Root Mean Squared Error between two images."""
    try:
        return np.sqrt(mse(img1, img2))
    except Exception as e:
        print(f"Error calculating RMSE: {e}")
        return 0.0


def calculate_mae(img1, img2):
    """Calculate Mean Absolute Error between two images."""
    try:
        return np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64)))
    except Exception as e:
        print(f"Error calculating MAE: {e}")
        return 0.0


def calculate_nrmse(img1, img2):
    """Calculate Normalized Root Mean Squared Error between two images."""
    try:
        return nrmse(img1, img2)
    except Exception as e:
        print(f"Error calculating NRMSE: {e}")
        return 0.0


def calculate_correlation(img1, img2):
    """Calculate correlation coefficient between two images."""
    try:
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2
        
        # Flatten images
        img1_flat = img1_gray.flatten().astype(np.float64)
        img2_flat = img2_gray.flatten().astype(np.float64)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(img1_flat, img2_flat)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        return 0.0


def calculate_gradient_magnitude_similarity(img1, img2):
    """Calculate Gradient Magnitude Similarity between two images."""
    try:
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2
        
        # Calculate gradients
        grad1 = np.sqrt(filters.sobel_h(img1_gray)**2 + filters.sobel_v(img1_gray)**2)
        grad2 = np.sqrt(filters.sobel_h(img2_gray)**2 + filters.sobel_v(img2_gray)**2)
        
        # Calculate similarity
        numerator = 2 * grad1 * grad2 + 1e-8
        denominator = grad1**2 + grad2**2 + 1e-8
        gms = np.mean(numerator / denominator)
        
        return gms
    except Exception as e:
        print(f"Error calculating GMS: {e}")
        return 0.0


def calculate_edge_similarity(img1, img2):
    """Calculate Edge Similarity between two images."""
    try:
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2
        
        # Edge detection
        edges1 = cv2.Canny(img1_gray, 50, 150)
        edges2 = cv2.Canny(img2_gray, 50, 150)
        
        # Calculate similarity
        intersection = np.logical_and(edges1, edges2)
        union = np.logical_or(edges1, edges2)
        
        if np.sum(union) == 0:
            return 1.0  # Both images have no edges
        
        return np.sum(intersection) / np.sum(union)
    except Exception as e:
        print(f"Error calculating Edge Similarity: {e}")
        return 0.0


def extract_metrics(reference_path, test_path):
    """Extract comprehensive quality metrics from two videos."""
    print(f"Analyzing: {os.path.basename(test_path)} vs {os.path.basename(reference_path)}")
    
    # Open video files
    ref_cap = cv2.VideoCapture(reference_path)
    test_cap = cv2.VideoCapture(test_path)
    
    if not ref_cap.isOpened():
        print(f"Error: Could not open reference video {reference_path}")
        return None
    
    if not test_cap.isOpened():
        print(f"Error: Could not open test video {test_path}")
        return None
    
    # Get video properties
    width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = test_cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
    
    # Initialize metrics storage
    metrics = {
        'psnr': [],
        'ssim': [],
        'mse': [],
        'rmse': [],
        'mae': [],
        'nrmse': [],
        'correlation': [],
        'gms': [],
        'edge_similarity': []
    }
    
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
        
        # Calculate all metrics
        metrics['psnr'].append(calculate_psnr(ref_frame, test_frame))
        metrics['ssim'].append(calculate_ssim(ref_frame, test_frame))
        metrics['mse'].append(calculate_mse(ref_frame, test_frame))
        metrics['rmse'].append(calculate_rmse(ref_frame, test_frame))
        metrics['mae'].append(calculate_mae(ref_frame, test_frame))
        metrics['nrmse'].append(calculate_nrmse(ref_frame, test_frame))
        metrics['correlation'].append(calculate_correlation(ref_frame, test_frame))
        metrics['gms'].append(calculate_gradient_magnitude_similarity(ref_frame, test_frame))
        metrics['edge_similarity'].append(calculate_edge_similarity(ref_frame, test_frame))
        
        frame_num += 1
        pbar.update(1)
    
    # Clean up
    pbar.close()
    ref_cap.release()
    test_cap.release()
    
    # Convert to numpy arrays for easier manipulation
    for key in metrics:
        metrics[key] = np.array(metrics[key])
    
    # Print summary statistics
    if len(metrics['psnr']) > 0:
        print(f"Quality Metrics Summary for {os.path.basename(test_path)}:")
        for metric_name, values in metrics.items():
            avg_val = np.mean(values)
            min_val = np.min(values)
            max_val = np.max(values)
            std_val = np.std(values)
            
            if metric_name == 'psnr':
                print(f"  {metric_name.upper()}: {avg_val:.2f} ± {std_val:.2f} dB (range: {min_val:.2f} - {max_val:.2f})")
            elif metric_name in ['mse', 'rmse', 'mae']:
                print(f"  {metric_name.upper()}: {avg_val:.4f} ± {std_val:.4f} (range: {min_val:.4f} - {max_val:.4f})")
            else:
                print(f"  {metric_name.upper()}: {avg_val:.4f} ± {std_val:.4f} (range: {min_val:.4f} - {max_val:.4f})")
        print()
    
    return metrics


def plot_metrics(reference_path, test_videos, output_path):
    """Generate comprehensive plots comparing multiple test videos against reference."""
    
    # Extract metrics for each test video
    all_metrics = {}
    
    for test_video in test_videos:
        video_name = os.path.basename(test_video).replace('.mp4', '')
        metrics = extract_metrics(reference_path, test_video)
        
        if metrics is not None:
            all_metrics[video_name] = metrics
    
    if not all_metrics:
        print("Error: No valid metrics extracted from any video")
        return False
    
    # Create comprehensive plots
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'Video Quality Metrics Comparison vs {os.path.basename(reference_path)}', fontsize=16, fontweight='bold')
    
    # Color palette for different videos
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Define plot configurations
    plot_configs = [
        ('psnr', 'PSNR (dB)', 'Higher is better', True),
        ('ssim', 'SSIM', 'Higher is better', True),
        ('mse', 'MSE', 'Lower is better', False),
        ('rmse', 'RMSE', 'Lower is better', False),
        ('mae', 'MAE', 'Lower is better', False),
        ('nrmse', 'NRMSE', 'Lower is better', False),
        ('correlation', 'Correlation', 'Higher is better', True),
        ('gms', 'Gradient Magnitude Similarity', 'Higher is better', True),
        ('edge_similarity', 'Edge Similarity', 'Higher is better', True)
    ]
    
    for idx, (metric_name, ylabel, description, higher_better) in enumerate(plot_configs):
        ax = axes[idx // 3, idx % 3]
        
        ax.set_title(f'{ylabel}\n({description})', fontsize=10, fontweight='bold')
        ax.set_xlabel('Frame Number')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        for i, (video_name, metrics) in enumerate(all_metrics.items()):
            if metric_name in metrics:
                frames = np.arange(len(metrics[metric_name]))
                color = colors[i % len(colors)]
                ax.plot(frames, metrics[metric_name], label=video_name, color=color, linewidth=1.5)
                
                # Add average line
                avg_val = np.mean(metrics[metric_name])
                ax.axhline(y=avg_val, color=color, linestyle='--', alpha=0.7, 
                          label=f'{video_name} avg: {avg_val:.4f}')
        
        ax.legend(fontsize=8)
        
        # Set y-axis limits based on metric type
        if metric_name in ['ssim', 'correlation', 'gms', 'edge_similarity']:
            ax.set_ylim(0, 1)
        elif metric_name == 'psnr':
            ax.set_ylim(bottom=0)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive quality metrics plots saved to: {output_path}")
    
    # Also save detailed metrics to CSV
    csv_path = output_path.replace('.png', '_detailed_metrics.csv')
    with open(csv_path, 'w') as f:
        # Write header
        header = "Video,Frame,PSNR,SSIM,MSE,RMSE,MAE,NRMSE,Correlation,GMS,Edge_Similarity\n"
        f.write(header)
        
        # Write data
        for video_name, metrics in all_metrics.items():
            num_frames = len(metrics['psnr'])
            for frame in range(num_frames):
                row = f"{video_name},{frame}"
                for metric_name in ['psnr', 'ssim', 'mse', 'rmse', 'mae', 'nrmse', 'correlation', 'gms', 'edge_similarity']:
                    if metric_name in metrics:
                        row += f",{metrics[metric_name][frame]:.6f}"
                    else:
                        row += ",N/A"
                f.write(row + "\n")
    
    print(f"Detailed metrics data saved to: {csv_path}")
    
    # Create a summary statistics table
    summary_path = output_path.replace('.png', '_summary.csv')
    with open(summary_path, 'w') as f:
        f.write("Video,Metric,Mean,Std,Min,Max\n")
        for video_name, metrics in all_metrics.items():
            for metric_name, values in metrics.items():
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                f.write(f"{video_name},{metric_name},{mean_val:.6f},{std_val:.6f},{min_val:.6f},{max_val:.6f}\n")
    
    print(f"Summary statistics saved to: {summary_path}")
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
    
    # Ensure that all outputs are stored inside an "evaluation" folder in the current directory
    evaluation_dir = os.path.join(os.getcwd(), "evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)

    # Override user-specified path to always reside in the evaluation directory
    output_filename = os.path.basename(args.output)
    args.output = os.path.join(evaluation_dir, output_filename)
    
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

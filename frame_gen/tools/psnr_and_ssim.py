# PSNR + SSIM Side-by-Side Plot
# NOTE: both images/frames should always have the same width x height (e.g., frame 1: 1920x1080 vs frame 2: 1920x1080). Otherwise, PSNR and SSIM do not work!!

import os
import sys
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import datetime
import numpy as np

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Paths to folders
current_dir = os.getcwd()
method = 'interpolation'

# Experiment 1
# original_folder = os.path.join(current_dir, method, 'original_frames_1920_1080')
# comparison_folder = os.path.join(current_dir, method, 'upscaled_original_frames_from_1280_720_to_1920_1080')

original_folder = os.path.join(current_dir, method, 'downscaled_original_frames_from_1920_1080_to_1280_720')
comparison_folder = os.path.join(current_dir, method, 'processed_frames_rife_1280_720')

# Get the list of files in both folders
original_files = sorted(os.listdir(original_folder))
comparison_files = sorted(os.listdir(comparison_folder))

# Print how many files are in each folder
print(f"Original folder has {len(original_files)} files.")
print(f"Comparison folder has {len(comparison_files)} files.")

# Check if the two folders have the same number of files and the names match
matching_files = [f for f in original_files if f in comparison_files]
print(f"Found {len(matching_files)} matching files.")

# Now, proceed with processing only the matching files
psnr_scores = []
ssim_scores = []
frame_labels = []

print("matching files:" + str(matching_files))

for filename in matching_files:
    original_path = os.path.join(original_folder, filename)
    comparison_path = os.path.join(comparison_folder, filename)

    img1 = cv2.imread(original_path)
    img2 = cv2.imread(comparison_path)

    if img1 is None or img2 is None:
        print(f"Unreadable: {filename}")
        continue

    if img1.shape != img2.shape:
        print(f"Shape mismatch: {filename}")
        continue

    print(f"Processing: {filename}")

    # Convert to grayscale for SSIM
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Skip PSNR = inf (i.e., MSE = 0) # With this verification of MSE, we can skip identical images! Because if they are identical PSNR goes to INFINITE!
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        print(f"Skipping perfect match (PSNR = INF): {filename}")
        continue

    # --- MODIFIED: Moved after MSE check
    psnr_score = psnr(img1, img2)
    ssim_score, _ = ssim(img1_gray, img2_gray, full=True)

    psnr_scores.append(psnr_score)
    ssim_scores.append(ssim_score)
    frame_labels.append(filename)

# After the loop, print averages
if psnr_scores:
    print(f"\nAverage PSNR: {sum(psnr_scores) / len(psnr_scores):.2f} dB")
if ssim_scores:
    print(f"Average SSIM: {sum(ssim_scores) / len(ssim_scores):.4f}")

x = np.arange(len(frame_labels))  # X-axis: frame indices or names

fig, ax1 = plt.subplots(figsize=(12, 6))

# PSNR on left Y-axis
color = 'tab:blue'
ax1.set_xlabel('Frame')
ax1.set_ylabel('PSNR (dB)', color=color)
ax1.plot(x, psnr_scores, color=color, label='PSNR')
ax1.tick_params(axis='y', labelcolor=color)

# SSIM on right Y-axis
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('SSIM', color=color)
ax2.plot(x, ssim_scores, color=color, label='SSIM')
ax2.tick_params(axis='y', labelcolor=color)

# Add average values to title
plt.title(f'PSNR and SSIM per Frame\nAvg PSNR: {sum(psnr_scores)/len(psnr_scores):.2f} dB | Avg SSIM: {sum(ssim_scores)/len(ssim_scores):.4f}')

# Save with timestamp
plt.tight_layout()
plot_name = f"psnr_ssim_plot_{timestamp}.png"
plt.savefig(plot_name, dpi=300)
print(f"Plot saved as {plot_name}")

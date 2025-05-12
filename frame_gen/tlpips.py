import os
import torch
import lpips
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# LPIPS (Learned Perceptual Image Patch Similarity) ranges from 0 (identical) to 1 (very different)

# === CONFIG ===
original_folder = 'interpolation/downscaled_original_frames_from_1920_1080_to_1280_720'
comparison_folder = 'interpolation/processed_frames_rife_1280_720'
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# === Set device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load LPIPS model ===
loss_fn = lpips.LPIPS(net='alex').to(device)

# === Image transform ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Optional: for performance, you can change or remove
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# === Get sorted list of frames ===
original_frames = sorted([f for f in os.listdir(original_folder) if f.endswith('.png')])
comparison_frames = sorted([f for f in os.listdir(comparison_folder) if f.endswith('.png')])

num_pairs = min(len(original_frames), len(comparison_frames)) - 1
print(f"Found {num_pairs + 1} frames in each folder. Calculating tLPIPS on {num_pairs} pairs...")

tlpips_values = []
frame_indices = []

for i in range(num_pairs):
    orig_0 = Image.open(os.path.join(original_folder, original_frames[i])).convert('RGB')
    orig_1 = Image.open(os.path.join(original_folder, original_frames[i+1])).convert('RGB')

    gen_0 = Image.open(os.path.join(comparison_folder, comparison_frames[i])).convert('RGB')
    gen_1 = Image.open(os.path.join(comparison_folder, comparison_frames[i+1])).convert('RGB')

    orig_0 = transform(orig_0).unsqueeze(0).to(device)
    orig_1 = transform(orig_1).unsqueeze(0).to(device)
    gen_0 = transform(gen_0).unsqueeze(0).to(device)
    gen_1 = transform(gen_1).unsqueeze(0).to(device)

    # Compute LPIPS distances
    lpips_orig = loss_fn(orig_0, orig_1).item()
    lpips_gen = loss_fn(gen_0, gen_1).item()

    # Temporal LPIPS
    t_lpips = abs(lpips_orig - lpips_gen)
    tlpips_values.append(t_lpips)
    frame_indices.append(i)

    print(f"[{i:04d}] LPIPS_orig={lpips_orig:.4f}, LPIPS_gen={lpips_gen:.4f}, tLPIPS={t_lpips:.4f}")

# === Average result ===
avg_tlpips = np.mean(tlpips_values)
print(f"\nAverage tLPIPS: {avg_tlpips:.4f}")

# === Plot ===
plt.figure(figsize=(12, 5))
plt.plot(frame_indices, tlpips_values, marker='o', linestyle='-', color='blue')
plt.title(f'tLPIPS per Frame Pair\nAverage tLPIPS: {avg_tlpips:.4f}')
plt.xlabel('Frame Index')
plt.ylabel('tLPIPS')
plt.grid(True)
plt.tight_layout()

plot_name = f"tLPIPS_plot_{timestamp}.png"
plt.savefig(plot_name, dpi=300)
plt.show()

print(f"Plot saved as {plot_name}")

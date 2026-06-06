import cv2
import os
import subprocess
import csv
from skimage.metrics import structural_similarity as _skimage_ssim

def ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return _skimage_ssim(gray1, gray2, data_range=255)

def psnr(img1, img2):
    return cv2.PSNR(img1, img2)

def vmaf_score(ref_path, tgt_path):
    cmd = [
        "ffmpeg", "-i", tgt_path, "-i", ref_path,
        "-lavfi", "[0:v][1:v]libvmaf", "-f", "null", "-"
    ]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    for line in result.stderr.splitlines():
        if "VMAF score" in line:
            try:
                return float(line.split(":")[-1].strip())
            except ValueError:
                return None
    return None

def mask_qr_with_ref(img_ref, img_tgt, qr_size=200, padding=10):
    h, w = img_ref.shape[:2]
    img_tgt_masked = img_tgt.copy()
    # Replace QR region in target with reference pixels
    img_tgt_masked[h-qr_size-padding:h-padding, w-qr_size-padding:w-padding] = img_ref[h-qr_size-padding:h-padding, w-qr_size-padding:w-padding]
    return img_tgt_masked


def mask_qr(img, qr_size=200, padding=10):
    h, w = img.shape[:2]
    img_masked = img.copy()
    img_masked[h-qr_size-padding:h-padding, w-qr_size-padding:w-padding] = 0
    return img_masked

def compare_images(ref_folder, tgt_folder, start_num, end_num, csv_path, qr_size=200, padding=10):
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "SSIM", "PSNR", "VMAF"])
        for i in range(start_num, end_num + 1):
            fname = f"{i:04d}.png"
            ref_img_path = os.path.join(ref_folder, fname)
            tgt_img_path = os.path.join(tgt_folder, fname)
            print(f"Processing {fname}...")
            if not os.path.exists(ref_img_path) or not os.path.exists(tgt_img_path):
                print(f"Missing: {ref_img_path} or {tgt_img_path}")
                continue
            img1 = cv2.imread(ref_img_path)
            img2 = cv2.imread(tgt_img_path)
            if img1 is None or img2 is None:
                print(f"Image load error for {fname}")
                continue
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            # Mask QR code region in both images
            img1_masked = mask_qr(img1, qr_size, padding)
            #img2_masked = mask_qr(img2, qr_size, padding)
            img2_masked = mask_qr_with_ref(img1, img2, qr_size=200, padding=10)
            ssim_val = ssim(img1_masked, img2_masked)
            psnr_val = psnr(img1_masked, img2_masked)
            # Save masked images as temp mp4 for VMAF
            cv2.imwrite("ref_tmp.png", img1_masked)
            cv2.imwrite("tgt_tmp.png", img2_masked)
            subprocess.run([
                "ffmpeg", "-y", "-framerate", "1", "-i", "ref_tmp.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "ref_tmp.mp4"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run([
                "ffmpeg", "-y", "-framerate", "1", "-i", "tgt_tmp.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "tgt_tmp.mp4"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            vmaf_val = vmaf_score("ref_tmp.mp4", "tgt_tmp.mp4")
            writer.writerow([fname, ssim_val, psnr_val, vmaf_val])
    # Clean up temp files
    for f in ["ref_tmp.png", "tgt_tmp.png", "ref_tmp.mp4", "tgt_tmp.mp4"]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    compare_images(
        "/home/alireza/mycg/CGReplay/server/Kombat",
        "/home/alireza/mycg/CGReplay/player/received_frames",
        2, 101,
        "/home/alireza/mycg/CGReplay/tools/VQ/videoQ.csv"
    )
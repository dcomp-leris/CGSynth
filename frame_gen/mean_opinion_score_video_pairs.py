import csv
import random
import string
from datetime import datetime

# === CONFIG ===
video_pairs = [
    ("scene_01_real.mp4", "scene_01_interp.mp4")#,
    #("scene_02_real.mp4", "scene_02_interp.mp4"),
    # Add more pairs as needed
]

output_csv = f'mos_results_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'

# === Create CSV header ===
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'scene', 'video_A_filename', 'video_B_filename', 
                     'A_score', 'B_score', 'A_comment', 'B_comment', 
                     'video_A_is_real', 'video_B_is_real'])

# Generate a random user_id with letters and digits (alphanumeric, length 10)
user_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

print(f"Generated User ID: {user_id}")

for real_vid, interp_vid in video_pairs:
    scene_name = real_vid.split('_real')[0]

    # Randomize order
    videos = [("A", real_vid, "real"), ("B", interp_vid, "synthetic")]
    random.shuffle(videos)

    print(f"\n=== Evaluate Scene: {scene_name} ===")
    print(f"Please watch the following videos in order and rate each 1–5 based on visual quality:")
    print(f"[Video A] => Video A")
    print(f"[Video B] => Video B")

    input("Press Enter after watching both videos...")

    # Get scores and comments
    a_score = input("MOS for Video A (1–5): ").strip()
    a_comment = input("Optional comment for Video A: ").strip()
    if not a_comment:
        a_comment = "N/A"

    b_score = input("MOS for Video B (1–5): ").strip()
    b_comment = input("Optional comment for Video B: ").strip()
    if not b_comment:
        b_comment = "N/A"

    # Write results
    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            user_id, scene_name,
            videos[0][1], videos[1][1],  # filenames
            a_score, b_score,
            a_comment, b_comment,
            videos[0][2] == "real", videos[1][2] == "real"  # boolean
        ])

    print("Ratings recorded.")

print(f"\nAll ratings saved to {output_csv}")

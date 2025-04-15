import os
import yaml
from pyzbar.pyzbar import decode
from PIL import Image, ImageDraw, ImageFont
import datetime

# Load configuration from YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

frames_folder = config["frames_folder"]
output_file = config["output_file"]
output_folder = config["output_folder"]

def extract_qr_data(image_path):
    """Extract QR code data from the image."""
    img = Image.open(image_path)
    decoded_objects = decode(img)

    # Iterate through decoded QR codes and return data and bounding box
    for obj in decoded_objects:
        qr_data = obj.data.decode("utf-8")
        qr_box = obj.rect  # Bounding box of the QR code
        return qr_data, qr_box
    return None, None

def remove_qr_and_add_text(img, qr_box, frame_name, output_folder):
    """Remove QR code from the image and add human-readable ID and timestamp."""
    draw = ImageDraw.Draw(img)

    # Remove QR code (fill the area with white)
    if qr_box:
        draw.rectangle(
            [(qr_box.left, qr_box.top), (qr_box.left + qr_box.width, qr_box.top + qr_box.height)],
            fill="white"
        )

    # Add human-readable ID (frame name) and timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"Frame: {frame_name}, Time: {timestamp}"

    # Choose a font size and position for the text
    font = ImageFont.load_default()
    text_position = (10, 10)  # Top-left corner

    # Add text to the image
    draw.text(text_position, text, font=font, fill="black")

    # Save the modified image to the output folder
    output_path = os.path.join(output_folder, frame_name)
    img.save(output_path)

def process_frames(frames_folder, output_file, output_folder):
    """Read each frame, extract QR code data, and save it to a log file."""
    with open(output_file, 'w') as out_f:
        # Ensure output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Process all the frames in the folder
        for frame_name in sorted(os.listdir(frames_folder)):
            if frame_name.endswith(".png"):
                frame_path = os.path.join(frames_folder, frame_name)

                # Extract QR code data from the image
                qr_data, qr_box = extract_qr_data(frame_path)

                img = Image.open(frame_path)

                if qr_data:
                    out_f.write(f"Frame: {frame_name}, QR Data: {qr_data}\n")
                    print(f"Frame: {frame_name}, QR Data: {qr_data}")

                    # Remove QR code and add human-readable ID and timestamp
                    remove_qr_and_add_text(img, qr_box, frame_name, output_folder)
                else:
                    print(f"No QR code found in {frame_name}")

    print(f"Data written to {output_file}")

# Call the function to process the frames
process_frames(frames_folder, output_file, output_folder)

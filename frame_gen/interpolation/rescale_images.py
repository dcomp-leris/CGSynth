import os
import cv2
import sys

original_folder = 'original_frames_1920_1080'
original_res = [1920, 1080]
target_downscaled_res = [1280, 720]
target_upscaled_res = [1920, 1080]
downscaled_folder = 'downscaled_original_frames_from_' + str(original_res[0]) + '_' + str(original_res[1]) + '_to_' + str(target_downscaled_res[0]) + '_' + str(target_downscaled_res[1])
upscaled_folder = 'upscaled_original_frames_from_' + str(target_downscaled_res[0]) + '_' + str(target_downscaled_res[1]) + '_to_' + str(target_upscaled_res[0]) + '_' + str(target_upscaled_res[1])


def downscale_images():
    os.makedirs(downscaled_folder, exist_ok=True)
    image_files = [f for f in os.listdir(original_folder) if f.endswith('.png')]
    
    for image_file in image_files:
        image_path = os.path.join(original_folder, image_file)
        image = cv2.imread(image_path)

        # Resize to target downscaled resolution
        image_resized = cv2.resize(image, target_downscaled_res, interpolation=cv2.INTER_AREA)
        
        # Save downscaled image
        downscaled_path = os.path.join(downscaled_folder, image_file)
        cv2.imwrite(downscaled_path, image_resized)

    print(f"Downscaled {len(image_files)} images to {target_downscaled_res}.")


def upscale_images():
    # Check if downscaled folder exists
    if not os.path.exists(downscaled_folder):
        print(f"[ERROR] Folder '{downscaled_folder}' does not exist.")
        print("Please downscale the images first or provide the expected resolution by editing 'target_downscaled_res'.")
        sys.exit(0)

    os.makedirs(upscaled_folder, exist_ok=True)
    image_files = [f for f in os.listdir(original_folder) if f.endswith('.png')]
    
    for image_file in image_files:
        #original_path = os.path.join(original_folder, image_file)
        downscaled_path = os.path.join(downscaled_folder, image_file)
        
        # Load original for size reference
        #original_image = cv2.imread(original_path)
        #original_shape = (original_image.shape[1], original_image.shape[0])  # width, height
        upscaled_shape = (target_upscaled_res[0], target_upscaled_res[1])

        # Load the downscaled image
        if not os.path.exists(downscaled_path):
            print(f"Skipped {image_file}: Downscaled version not found.")
            continue

        downscaled_image = cv2.imread(downscaled_path)

        # Resize back to original shape
        image_upscaled = cv2.resize(downscaled_image, upscaled_shape, interpolation=cv2.INTER_CUBIC)
        
        # Save upscaled image
        upscaled_path = os.path.join(upscaled_folder, image_file)
        cv2.imwrite(upscaled_path, image_upscaled)

    print(f"Upscaled {len(image_files)} images back to original resolution.")


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Downscale images")
    print("2. Upscale images from downscaled versions")
    choice = input("Enter 1 or 2: ").strip()

    if choice == '1':
        downscale_images()
    elif choice == '2':
        upscale_images()
    else:
        print("Invalid choice. Please enter 1 or 2.")

    cv2.destroyAllWindows()

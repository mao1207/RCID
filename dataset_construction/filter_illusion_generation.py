from PIL import Image
import os
import numpy as np
import cv2

def rgb_to_hsv(image):
    """Convert an RGB image to HSV color space using OpenCV."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)

def hsv_to_rgb(hsv_image):
    """Convert an HSV image to RGB color space using OpenCV."""
    return Image.fromarray(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB))

def apply_contrast_filter(image_path, output_path, target_color, contrast_color, tolerance=30):
    """
    Apply a contrast filter in HSV space to suppress a target color and apply a new filter color.
    
    Parameters:
    - image_path: Path to the source image.
    - output_path: Path where the processed image will be saved.
    - target_color: The color (Cd) to suppress (e.g., 'red', 'blue', 'yellow').
    - contrast_color: The contrasting filter color (Cf) to overlay.
    - tolerance: The range of HSV values around the target color to suppress (default 30).
    """
    # Open the image and convert to HSV color space
    image = Image.open(image_path).convert("RGB")
    hsv_image = rgb_to_hsv(image)

    # Define the target and contrasting colors in HSV space
    target_hue_ranges = {
        'red': (0, 15),  # Red in HSV is around 0-15 degrees
        'blue': (90, 150),  # Blue in HSV is around 90-150 degrees
        'yellow': (45, 75)  # Yellow in HSV is around 45-75 degrees
    }

    # Get the hue range for the target color
    target_hue_range = target_hue_ranges.get(target_color.lower())
    if target_hue_range is None:
        raise ValueError(f"Unsupported target color: {target_color}")

    # Define the contrasting color's hue
    contrast_hues = {
        'red': 120,  # Green is opposite red on the color wheel
        'blue': 30,  # Orange is opposite blue
        'yellow': 180  # Cyan is opposite yellow
    }

    contrast_hue = contrast_hues.get(contrast_color.lower())
    if contrast_hue is None:
        raise ValueError(f"Unsupported contrast color: {contrast_color}")

    # Iterate over all pixels in the HSV image
    hsv_image = hsv_image.astype(np.float32)

    for i in range(hsv_image.shape[0]):
        for j in range(hsv_image.shape[1]):
            # Extract the pixel's HSV values
            h, s, v = hsv_image[i, j]
            
            # Check if the pixel's hue is within the target color's range
            if target_hue_range[0] <= h <= target_hue_range[1]:
                # Suppress the target color by shifting the hue to the contrast color
                hsv_image[i, j, 0] = contrast_hue  # Shift hue to the opposite color
                hsv_image[i, j, 1] = max(s * 1.5, 255)  # Increase saturation slightly
                hsv_image[i, j, 2] = min(v * 1.1, 255)  # Slightly increase brightness
            else:
                # Leave pixels outside the target color unchanged
                pass

    # Convert the modified HSV image back to RGB
    hsv_image = np.clip(hsv_image, 0, 255).astype(np.uint8)
    output_image = hsv_to_rgb(hsv_image)

    # Save the result
    output_image.save(output_path)

def process_images_in_folder(src_folder, dest_folder, target_color, contrast_color, tolerance=30):
    """
    Processes all images in a source folder and applies the contrast filter to each image.
    
    Parameters:
    - src_folder: The folder containing the images to process.
    - dest_folder: The folder where the processed images will be saved.
    - target_color: The color to suppress (e.g., 'red', 'blue', 'yellow').
    - contrast_color: The contrasting filter color to overlay (e.g., 'green', 'orange', 'cyan').
    - tolerance: The range of hues around the target color to suppress.
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Iterate through all files in the source folder
    for filename in os.listdir(src_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(src_folder, filename)
            output_path = os.path.join(dest_folder, filename)
            
            # Apply the contrast filter to the image
            apply_contrast_filter(image_path, output_path, target_color, contrast_color, tolerance)

# Source and destination folder paths
src_folder = '/path/to/source/folder'
dest_folder = '/path/to/destination/folder'

# Target and contrast colors (e.g., suppress red, apply green filter)
target_color = 'red'  # Can be 'red', 'blue', or 'yellow'
contrast_color = 'green'  # Can be 'green', 'orange', or 'cyan'

# Call the function to process images
process_images_in_folder(src_folder, dest_folder, target_color, contrast_color)

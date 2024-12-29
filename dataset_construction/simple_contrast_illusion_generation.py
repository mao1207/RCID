import matplotlib.pyplot as plt
import numpy as np
import os
import random
import json

def color_distance(c1, c2):
    """Calculate the 'distance' between two colors to ensure they are distinct.
    
    Args:
        c1 (str): Hex code of the first color.
        c2 (str): Hex code of the second color.
    
    Returns:
        int: A value representing the difference between the two colors.
    """
    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)
    return abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple.
    
    Args:
        hex_color (str): Hex code of the color.
    
    Returns:
        tuple: RGB representation of the hex color.
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def adjust_brightness(hex_color, factor):
    """Adjust the brightness of a color by a given factor.
    
    Args:
        hex_color (str): Hex code of the color.
        factor (float): Brightness factor, greater than 1 to lighten, less than 1 to darken.
    
    Returns:
        str: The hex code of the adjusted color.
    """
    rgb = hex_to_rgb(hex_color)
    brighter = tuple(min(255, int(col * factor)) for col in rgb)
    return "#{:02x}{:02x}{:02x}".format(*brighter)

def generate_random_color():
    """Generates a random hex color.
    
    Returns:
        str: Random hex color code.
    """
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def generate_color_palette():
    """Generates two distinct random colors and their brightness variants.
    
    Returns:
        list: A list of four colors including two main colors and two brightness variants.
    """
    color1 = generate_random_color()

    if random.random() < 0.5:
        color2 = adjust_brightness(color1, 1.0)
        color1 = adjust_brightness(color1, 1.0)
    else:
        color2 = adjust_brightness(color1, 1.0)
        color1 = adjust_brightness(color1, 1.0)
    
    color3 = generate_random_color()
    while color_distance(color3, color1) < 256 or color_distance(color3, color2) < 256:
        color3 = generate_random_color()

    lighter_factor = 1 + random.uniform(0.05, 0.10)
    darker_factor = 1 - random.uniform(0.05, 0.10)

    lighter_color = adjust_brightness(color3, lighter_factor)
    darker_color = adjust_brightness(color3, darker_factor)

    return [color1, color2, lighter_color, darker_color]

def generate_height():
    """Generates random heights for the blocks with a difference between 4 and 7.
    
    Returns:
        tuple: Low and high heights for the blocks.
    """
    while True:
        num1 = random.randint(0, 10)
        num2 = random.randint(0, 10)
        if abs(num1 - num2) >= 4 and abs(num1 - num2) <= 7:
            return min(num1, num2), max(num1, num2)
        
def generate_width():
    """Generates random widths for the blocks with a difference of at least 4.
    
    Returns:
        tuple: Left and right positions for the blocks.
    """
    while True:
        num1 = random.randint(0, 5)
        num2 = random.randint(0, 5)
        if abs(num1 - num2) >= 4:
            return min(num1, num2), max(num1, num2)

def generate_batch_illusion_images(num_images, output_folder):
    """Generates a batch of illusion images with 500x500 resolution, based on color palette and geometric placements.
    
    Args:
        num_images (int): Number of images to generate.
        output_folder (str): Folder path where the generated images will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Image dimension and block placement parameters
    grid_height = 10  # Number of rows in grid
    grid_width = 16   # Number of columns in grid
    cell_size = 50    # Size of each grid cell in pixels
    image_height = grid_height * cell_size  # Total image height
    image_width = grid_width * cell_size    # Total image width
    
    # Dictionary to store rectangle centers for JSON output
    rectangle_centers = {}

    for i in range(num_images):
        # Generate random colors for the image
        deep_color, light_color, mid_color1, mid_color2 = generate_color_palette()

        # Create a blank grid with the light color
        image_grid = np.full((image_height, image_width, 3), hex_to_rgb(light_color), dtype=np.uint8)
        
        # Fill the left half of the grid with deep color
        for row in range(grid_height):
            for col in range(grid_width // 2):  # Only fill the left half
                start_row, start_col = row * cell_size, col * cell_size
                end_row, end_col = start_row + cell_size, start_col + cell_size
                image_grid[start_row:end_row, start_col:end_col] = hex_to_rgb(deep_color)

        # Randomly generate block positions for the illusion effect
        low, high = generate_height()
        left, right = generate_width()

        center_left_x = (left + right) / 2
        center_left_y = (low + high) / 2

        center_right_x = grid_width - center_left_x
        center_right_y = center_left_y

        rectangle_centers[f'illusion_image_{i+1}.png'] = {
            'center_left_x': center_left_x,
            'center_left_y': center_left_y,
            'center_right_x': center_right_x,
            'center_right_y': center_right_y,
            'bg_color': hex_to_rgb(mid_color1)
        }

        # Place the mid-tone blocks in specified areas
        blocks = [(low, left, high, right), (low, grid_width - right, high, grid_width - left)]
        for id, block in enumerate(blocks):
            start_row, start_col, end_row, end_col = [x * cell_size for x in block]
            color = mid_color1 if id == 0 else mid_color2
            image_grid[start_row:end_row, start_col:end_col] = hex_to_rgb(color)

        # Place surrounding blocks
        blocks_surrounding = [(low - 1, left - 1, high + 1, right + 1), (low - 1, grid_width - right, high + 1, grid_width - left)]
        for id, block in enumerate(blocks_surrounding):
            for row in range(block[0], block[2]):
                for col in range(block[1], block[3]):
                    if 0 <= row < grid_height and 0 <= col < grid_width and random.random() < 0.2:
                        start_row, start_col = row * cell_size, col * cell_size
                        end_row, end_col = start_row + cell_size, start_col + cell_size
                        color = mid_color1 if id == 0 else mid_color2
                        image_grid[start_row:end_row, start_col:end_col] = hex_to_rgb(color)
        
        # Save the generated image
        plt.figure(figsize=(13, 13))  # Scale up figure size for higher resolution
        plt.imshow(image_grid)
        plt.axis('off')
        plt.savefig(os.path.join(output_folder, f'illusion_image_{i+1}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

    # Save rectangle centers to a JSON file for future reference
    with open(os.path.join(output_folder, 'rectangle_centers.json'), 'w') as f:
        json.dump(rectangle_centers, f)

# Set parameters
num_images = 5000
output_folder = '/path/to/output/folder'  # Change to your desired output folder

# Generate images
generate_batch_illusion_images(num_images, output_folder)

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import matplotlib.transforms as transforms
import json
import colorsys

# Directions for stripe orientation
directions = ['horizontal', 'vertical', 'diagonal_1', 'diagonal_2']

def hex_to_rgb(hex_color):
    """Converts a hex color string to an RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def adjust_brightness(hex_color, factor):
    """Adjusts the brightness of a color by a given factor.
    A value > 1 brightens the color, < 1 darkens it.
    """
    rgb = hex_to_rgb(hex_color)
    brighter = tuple(min(255, int(col * factor)) for col in rgb)
    return "#{:02x}{:02x}{:02x}".format(*brighter)

def color_distance(c1, c2):
    """Calculates the 'distance' between two colors to ensure they are distinct.
    The distance is based on the difference in RGB values.
    """
    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)
    return abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)

def random_color():
    """Generates a random color with a hue, saturation, and lightness that ensures
    the color is neither too close to white nor too dark.
    """
    h = random.randint(0, 360)  # Random hue (color)
    s = random.randint(60, 100)  # High saturation
    l = random.randint(30, 70)   # Lightness between 30% and 70%
    
    # Convert HSL to RGB
    r, g, b = colorsys.hls_to_rgb(h/360.0, l/100.0, s/100.0)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

def save_results_to_json(image_name, direction, sample_num_bars, results):
    """Saves the question and options to a JSON file after generating a stripe pattern image.
    
    Parameters:
    - image_name: The name of the generated image.
    - direction: The stripe direction (horizontal, vertical, etc.).
    - sample_num_bars: The number of bars used in the generated image.
    - results: A list where the result will be appended.
    """
    if direction == "horizontal":
        question = "How do the stripe colors and brightness compare on the left and right stripes interleaved between black stripes?"
        options = ["Left side is darker", "Right side is darker", "They are about the same"]
    elif direction == "vertical":
        question = "How do the stripe colors and brightness compare on the top and bottom stripes interleaved between black stripes?"
        options = ["Top side is darker", "Bottom side is darker", "They are about the same"]
    elif direction == "diagonal_1":
        question = "How do the stripe colors and brightness compare on the top-left and bottom-right stripes interleaved between black stripes?"
        options = ["Top-left side is darker", "Bottom-right side is darker", "They are about the same"]
    elif direction == "diagonal_2":
        question = "How do the stripe colors and brightness compare on the bottom-left and top-right stripes interleaved between black stripes?"
        options = ["Bottom-left side is darker", "Top-right side is darker", "They are about the same"]

    randomized_options = random.sample(options, len(options))
    correct_index = randomized_options.index("They are about the same")  # Find index of correct answer
    
    answers = ['I', "II", "III"]
    
    # Prepare JSON entry
    entry = {
        "image_name": image_name,
        "question": question,
        "options": [f"I. {randomized_options[0]}",
                    f"II. {randomized_options[1]}",
                    f"III. {randomized_options[2]}"],
        "result": answers[correct_index],  # Correct answer index
        "num_bars": sample_num_bars
    }
    
    results.append(entry)  # Append entry to results list

def generate_image(color1, color2, background, direction, output_file):
    """Generates a stripe pattern image with two colors (color1, color2) and saves it.
    Supports horizontal, vertical, and diagonal orientations.
    
    Parameters:
    - color1: The color for the first set of stripes.
    - color2: The color for the second set of stripes.
    - background: Background color.
    - direction: The direction of the stripes (horizontal, vertical, diagonal_1, diagonal_2).
    - output_file: Path to save the generated image.
    
    Returns:
    - sample_num_bars: Number of bars used in the image.
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(30, 18))
    fig.patch.set_facecolor(background)  # Set background color

    # Set up bar height, gap, and margin
    bar_height = 1
    gap = 1
    margin = 1

    # For horizontal direction
    if direction == 'horizontal':
        num_bars = [5, 6, 7, 8]
        first_position = random.choice([1.5, 7])
        second_position = 7 if first_position == 1.5 else 1.5

        left_curve = random.random() < 0.5
        right_curve = random.random() < 0.5

        sample_num_bars = random.choice(num_bars)
        for i in range(sample_num_bars):
            y_position = i * (bar_height + gap) + margin
            ax.add_patch(plt.Rectangle((margin, y_position), 12, bar_height, color=color1))
            ax.add_patch(plt.Rectangle((first_position + left_curve * random.uniform(-0.4, 0.4) + margin, y_position), 3.5, bar_height, color=color2))
            if i != sample_num_bars - 1:
                ax.add_patch(plt.Rectangle((second_position + right_curve * random.uniform(-0.4, 0.4) + margin, y_position + gap), 3.5, gap, color=color2))
            if i == sample_num_bars - 1:
                if random.random() < 0.5:
                    ax.add_patch(plt.Rectangle((second_position + right_curve * random.uniform(-0.4, 0.4) + margin, y_position + gap), 3.5, gap, color=color2))
            
        ax.set_xlim(0, 12 + 2 * margin)
        ax.set_ylim(0, (sample_num_bars * (bar_height + gap)) + 2 * margin)
        ax.axis('off')

    # For vertical direction
    elif direction == 'vertical':
        num_bars = [10, 11, 12, 13, 14]
        first_position = random.choice([1.0, 7])
        second_position = 7 if first_position == 1.0 else 1.0

        left_curve = random.random() < 0.5
        right_curve = random.random() < 0.5

        sample_num_bars = random.choice(num_bars)
        for i in range(sample_num_bars):
            x_position = i * (bar_height + gap) + margin
            ax.add_patch(plt.Rectangle((x_position, margin), bar_height, 12, color=color1))
            ax.add_patch(plt.Rectangle((x_position, first_position + left_curve * random.uniform(-0.4, 0.4) + margin), bar_height , 4.0, color=color2))
            if i != sample_num_bars - 1:
                ax.add_patch(plt.Rectangle((x_position + bar_height, second_position + right_curve * random.uniform(-0.4, 0.4) + margin), gap, 4.0, color=color2))
            if i == sample_num_bars - 1:
                if random.random() < 0.5:
                    ax.add_patch(plt.Rectangle((x_position + bar_height, second_position + right_curve * random.uniform(-0.4, 0.4) + margin), gap, 4.0, color=color2))

        ax.set_ylim(0, 12 + 2 * margin)
        ax.set_xlim(0, (sample_num_bars * (bar_height + gap)) + 2 * margin)
        ax.axis('off')

    # For diagonal directions
    elif direction == 'diagonal_1' or direction == 'diagonal_2':
        num_bars = [6, 7, 8, 9]
        first_position = random.choice([1.5, 9])
        second_position = 9 if first_position == 1.5 else 1.5

        left_curve = random.random() < 0.5
        right_curve = random.random() < 0.5

        sample_num_bars = random.choice(num_bars)
        for i in range(sample_num_bars):
            y_position = i * (bar_height + gap) + margin
            ax.add_patch(plt.Rectangle((margin, y_position), 15, bar_height, color=color1))
            ax.add_patch(plt.Rectangle((first_position + left_curve * random.uniform(-0.4, 0.4) + margin, y_position), 5, bar_height, color=color2))
            if i != sample_num_bars - 1:
                ax.add_patch(plt.Rectangle((second_position + right_curve * random.uniform(-0.4, 0.4) + margin, y_position + gap), 5, gap, color=color2))
            if i == sample_num_bars - 1:
                if random.random() > 0.5:
                    ax.add_patch(plt.Rectangle((second_position + right_curve * random.uniform(-0.4, 0.4) + margin, y_position + gap), 5, gap, color=color2))

        ax.set_xlim(0, 12 + 2 * margin)
        ax.set_ylim(0, (sample_num_bars * (bar_height + gap)) + 2 * margin)
        ax.axis('off')

    # Rotate diagonal patterns
    if direction == 'diagonal_1':
        trans = transforms.Affine2D().rotate_deg_around(10, 6, -45) + \
            transforms.Affine2D().translate(-2.12 - 0.11 * sample_num_bars, -1 + 0.15 * sample_num_bars) + ax.transData
        for patch in ax.patches:
            patch.set_transform(trans)
    if direction == 'diagonal_2':
        trans = transforms.Affine2D().rotate_deg_around(10, 6, 45) + \
            transforms.Affine2D().translate(-2.45 + 0.33 * sample_num_bars, 0.5 + 0.15 * sample_num_bars) + ax.transData
        for patch in ax.patches:
            patch.set_transform(trans)

    # Save the image
    plt.savefig(output_file, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    return sample_num_bars

# Initialize results list
results = []

# Generate and save images
for direction in directions:
    for i in range(50):  # Generate 50 images per direction
        output_dir = f'./generated_images_{direction}'  # Placeholder directory
        os.makedirs(output_dir, exist_ok=True)

        base_color = random_color()
        color1 = adjust_brightness(base_color, 0.1)
        color2 = adjust_brightness(base_color, 1.5)
        background = random_color()

        # Ensure background color is distinct from base color
        while color_distance(base_color, background) < 256:
            background = random_color()
        background = adjust_brightness(background, 0.35)

        output_file = os.path.join(output_dir, f'image_{i}_{direction}.png')
        sample_num_bars = generate_image(color1, color2, background, direction, output_file)

        # Save question and answer to JSON
        save_results_to_json(f'image_{i}_{direction}.png', direction, sample_num_bars, results)

# Save the results to a JSON file
with open('./train_results.json', 'w') as json_file:  # Placeholder path for saving results
    json.dump(results, json_file, indent=4)

print(f"Images and results saved.")

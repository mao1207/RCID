import os
import random
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import clip
import time

# Setting up seed for reproducibility
current_time = int(time.time())
random.seed(current_time)

# Model paths (replace with your paths)
base_model_path = "pt-sk/stable-diffusion-1.5"
controlnet_path = "mao1207/color-diffusion/color-diffusion-model"

# Device configuration (make sure you are using a compatible CUDA device)
device = "cuda:3"
mode_name = "your_mode_name"

def get_text_color(bg_color):
    """
    Determines the text color (black or white) based on the brightness of the background color.
    
    Args:
    bg_color (tuple): The RGB values of the background color.
    
    Returns:
    tuple: RGB values of the text color (either black or white).
    """
    brightness = (bg_color[0] * 299 + bg_color[1] * 587 + bg_color[2] * 114) / 1000
    return (255, 255, 255) if brightness < 128 else (0, 0, 0)

def add_labels_to_image(image_path, output_folder, file_name, center_left, center_right, bg_color):
    """
    Adds labels ('A' and 'B') to an image at specified positions (center_left, center_right).
    The text color is determined by the background color's brightness.
    
    Args:
    image_path (str): Path to the image file.
    output_folder (str): Folder where the output image will be saved.
    file_name (str): Name for the output image file.
    center_left (tuple): Coordinates for placing 'A' label.
    center_right (tuple): Coordinates for placing 'B' label.
    bg_color (tuple): RGB background color used to determine text color.
    """
    # Open image and prepare drawing context
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Load font for the label (adjust the font path and size as needed)
    font = ImageFont.truetype("path_to_your_font_file.ttf", 50)

    # Set text color based on background color brightness
    text_color = get_text_color(bg_color)

    # Add text ('A' and 'B') to image
    draw.text(center_left, "A", font=font, fill=text_color)
    draw.text(center_right, "B", font=font, fill=text_color)

    # Save the modified image
    save_image(img, output_folder, file_name)

def save_image(image, output_folder, file_name):
    """
    Saves an image to the specified folder with the provided file name.
    
    Args:
    image (PIL Image): The image to save.
    output_folder (str): Folder to save the image in.
    file_name (str): Name of the output file.
    
    Returns:
    str: Path to the saved image.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image.save(os.path.join(output_folder, file_name))
    return os.path.join(output_folder, file_name)

def load_texts_from_jsonl(file_path):
    """
    Loads texts from a JSONL file.
    
    Args:
    file_path (str): Path to the JSONL file.
    
    Returns:
    list: A list of text strings.
    """
    texts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # Parse JSONL file where each line is a JSON object
        json_objects = content.strip().split('}\n{')
        json_objects = [f'{{{obj}}}' for obj in json_objects]
        
        for obj in json_objects:
            try:
                data = json.loads(obj)
                texts.append(data['text'])
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e} in object: {obj}")
    return texts

def get_random_text(texts):
    """
    Returns a random text from the provided list of texts.
    
    Args:
    texts (list): List of text strings.
    
    Returns:
    str: A randomly selected text.
    """
    return random.choice(texts)

def get_all_images_from_folder(folder_path):
    """
    Retrieves all image paths from a given folder, shuffling them.
    
    Args:
    folder_path (str): Path to the folder containing images.
    
    Returns:
    list: List of image file paths.
    """
    images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)
    return images

# Initialize CLIP model for captioning
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

def find_best_caption(image_path, captions):
    """
    Finds the best caption for a given image by calculating similarity between the image and captions using CLIP.
    
    Args:
    image_path (str): Path to the image file.
    captions (list): List of captions to compare against.
    
    Returns:
    str: The best matching caption.
    """
    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(caption) for caption in captions]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text_inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T).squeeze()

    best_caption_index = similarity.argmax().item()
    return captions[best_caption_index]

# Paths to data (replace with your actual paths)
image_folder = "path_to_images_folder"
output_folder = "path_to_output_folder"
output_folder_with_labels = "path_to_output_with_labels_folder"

# Load all images and text prompts
images = get_all_images_from_folder(image_folder)
prompts = load_texts_from_jsonl("path_to_prompts_jsonl")

# Initialize models
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float32).to(device)
pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=controlnet, torch_dtype=torch.float32).to(device)

# Set up faster diffusion process
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

# Load positions (replace with your actual path)
positions = json.load(open("path_to_positions_file.json"))

image_info = {}

# Process each image
for image_path in images:
    control_image = load_image(image_path)
    
    for i in range(10):
        # Get a random prompt
        prompt = get_random_text(prompts)
        
        # Generate image using ControlNet
        generator = torch.manual_seed(random.randint(0, 100000))
        generated_image = pipe(
            prompt, num_inference_steps=30, generator=generator, image=control_image, controlnet_conditioning_scale=1.1
        ).images[0]
        
        # Save generated image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        extension = os.path.splitext(image_path)[1]
        output_file_name = f"{base_name}_{i+1}{extension}"
        generated_image_path = save_image(generated_image, output_folder, output_file_name)

        # Retrieve center positions and background color for text placement
        position_data = positions.get(f"{base_name}{extension}")
        if position_data:
            center_left = (position_data['center_left_x'], position_data['center_left_y'])
            center_right = (position_data['center_right_x'], position_data['center_right_y'])
            bg_color = position_data['bg_color']

            # Add labels to the generated image
            add_labels_to_image(generated_image_path, output_folder_with_labels, output_file_name, center_left, center_right, bg_color)

            # Store image info
            image_info[generated_image_path] = {"caption": prompt, "center_left": center_left, "center_right": center_right}

        # Save image info to JSON file
        with open(os.path.join(output_folder, 'image_info.json'), 'w') as f:
            json.dump(image_info, f)
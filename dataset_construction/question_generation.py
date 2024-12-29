import base64
import requests
import json
import os
import random
import time

# Define API Key and paths (replace these with your actual paths and API key)
api_key = "your_api_key_here"
few_shot_dir = '/path/to/fewshot/dataset/'
img_dir = '/path/to/images/dataset/'

# Function to encode an image to base64
def encode_image(image_path):
    """
    Encodes the image at the given path to a base64 string.
    
    Parameters:
    image_path (str): Path to the image file.

    Returns:
    str: Base64 encoded image string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to retrieve all image filenames from a folder
def get_images_from_folder(folder):
    """
    Retrieves all PNG image filenames from the specified folder,
    sorted by the numeric part of their filenames.

    Parameters:
    folder (str): Path to the folder containing images.

    Returns:
    list: Sorted list of image filenames.
    """
    images = [f for f in os.listdir(folder) if f.endswith('.png')]
    images.sort(key=lambda x: int(x.split('.')[0]))  # Sort by numeric part of the filename
    return images

# Example few-shot dataset (these are example questions related to images)
few_shot_examples = [
    {
        "image_path": "928.png",
        "question": "How does the shade of green on the wall directly behind the toilet compare to the shade of green on the wall on the top right? Do they have the same color and same brightness?"
    },
    {
        "image_path": "954.png",
        "question": "How do the color and brightness of the sand compare between the left side of the baseball player in the dark blue uniform and the right side of the baseball player in the white uniform?"
    },
    {
        "image_path": "536.png",
        "question": "Observe the two sections of refrigerators in the kitchen, one located towards the left and predominantly shaded, and the other to the right under more direct lighting. How do the colors and brightnesses of the half top part of these two refrigerator sections compare to each other?"
    },
    {
        "image_path": "2.png",
        "question": "How do the colors of the backrest cushion on the left side of the sofa compare to the color of the cushion on the right side of the dog?"
    },
    {
        "image_path": "70.png",
        "question": "How do the color and brightness of the purple background on the walls to the left and right of the green figure compare?"
    },
    {
        "image_path": "92.png",
        "question": "How do the color and brightness of the purple panel on the blue wall on the left compare to the purple panel in the darker area on the right?"
    },
    {
        "image_path": "6.png",
        "question": "How do the color and brightness of the purple toilet on the left compare to the purple towel placed on the toilet on the right?"
    },
]

# Encode few-shot example images to base64
encoded_few_shot_examples = []
for example in few_shot_examples:
    encoded_image = encode_image(os.path.join(few_shot_dir, example["image_path"]))
    encoded_few_shot_examples.append({
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": example["question"]
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            }
        ]
    })

# Retrieve image files from the image folder
images = get_images_from_folder(img_dir)

# Path to store the generated questions (replace with actual path)
json_path = '/path/to/output/questions.json'

# Load previously generated questions if the file exists
if os.path.exists(json_path):
    with open(json_path, 'r') as file:
        generated_questions = json.load(file)
else:
    generated_questions = []

# Set of processed image IDs to avoid reprocessing
processed_images = {q['image_id'] for q in generated_questions}

for image in images:
    # Skip image if it's already processed
    if image in processed_images:
        continue

    image_path = os.path.join(img_dir, image)

    # Encode the image to base64
    base64_image = encode_image(image_path)

    # Define headers for the OpenAI API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Payload for the OpenAI API request, including the few-shot examples
    payload = {
        "model": "gpt-4-turbo",
        "messages": encoded_few_shot_examples + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You will be given some images, and your task is to generate a question for each image that prompts participants to compare the colors of two distinct areas marked as A and B in the image. The areas are usually symmetrical, and their colors are similar. Please describe the differences between the areas in a clear and unambiguous manner, avoiding the use of labels such as 'A' and 'B'. Use directional and color terms to help the participants observe the areas."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    print(f"Processing image: {image_path}")

    # Request question generation from OpenAI API
    while True:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        try:
            question_text = response.json()['choices'][0]['message']['content']
            break
        except KeyError:
            print("No 'choices' key in the response, retrying...")
            time.sleep(1)  # Wait for 1 second before retrying

    # Generate options for the question
    options = [
        "left is darker",
        "They are exact same",
        "right is darker"
    ]
    random.shuffle(options)

    # Create a mapping of options to labels
    option_labels = ["A", "B", "C"]
    labeled_options = {label: option for label, option in zip(option_labels, options)}

    # Determine the correct answer (assuming "They are exact same")
    correct_answer_text = "They are exact same"
    correct_answer_label = next(label for label, option in labeled_options.items() if option == correct_answer_text)

    # Format the options for display
    formatted_options = "\n".join([f"{label}. {option}" for label, option in labeled_options.items()])

    # Add options to the question
    question_with_options = f"{question_text}\n\nHere are the options you can choose from:\n{formatted_options}"

    # Save the generated question along with the correct answer
    generated_question = {
        "image_id": image,
        "question": question_with_options,
        "correct_answer": correct_answer_label
    }
    
    generated_questions.append(generated_question)

    # Save the updated questions to the JSON file
    with open(json_path, 'w') as file:
        json.dump(generated_questions, file, indent=4)

    print(f"Question saved for image: {image}")

print("All questions generated and saved successfully.")

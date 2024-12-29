import os
import json

def main():
    # Define the paths for conditioning images, regular images, and the captions file
    conditioning_images_path = '/path/to/conditioning_images'
    images_path = '/path/to/images'
    captions_file_path = '/path/to/captions.json'
    
    # Path to store the output JSON file
    output_file_path = '/path/to/output_file.jsonl'

    # Load the captions data from the JSON file
    with open(captions_file_path, 'r') as file:
        captions_data = json.load(file)
    
    # Prepare a dictionary for quick caption lookup based on image_id
    captions_dict = {}
    for item in captions_data['annotations']:
        image_id = item['image_id']
        caption = item['caption']
        # Store the first caption found for each image_id
        if image_id not in captions_dict:
            captions_dict[image_id] = caption
    
    # Open the output file for writing
    num_processed = 0
    with open(output_file_path, 'w') as outfile:
        # Process each image in the conditioning images folder
        for filename in os.listdir(conditioning_images_path):
            if filename.endswith(".jpg"):
                # Extract the image ID from the filename, removing leading zeros and converting to int
                image_id = int(filename.split('.')[0].lstrip('0'))
                conditioning_image_path = os.path.join(conditioning_images_path, filename)
                image_path = os.path.join(images_path, filename)
                
                # Get the caption for this image, default to "No caption available" if none found
                caption = captions_dict.get(image_id, "No caption available")
                
                # Prepare the output JSON for each image
                output = {
                    "text": caption,
                    "image": f"images/{filename}",
                    "conditioning_image": f"conditioning_images/{filename}"
                }
                
                # Write the JSON data to the output file, each entry on a new line
                json.dump(output, outfile, indent=2)
                num_processed += 1
                outfile.write('\n')  # Ensure that each JSON object is on a new line

    # Output the total number of processed images
    print(f"Processed {num_processed} images.")

if __name__ == "__main__":
    main()

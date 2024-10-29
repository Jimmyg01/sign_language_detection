import os
import cv2
import numpy as np

# Set the path to your dataset
dataset_path = 'C:\\Users\\jimmy\\OneDrive\\Desktop\\sign_language_detection\\Data'

# Define the new image size
new_size = (128, 128)  # You can change this to (128, 128) if needed

# Loop through each number folder
for number in range(10):
    folder_path = os.path.join(dataset_path, str(number))
    if not os.path.exists(folder_path):
        continue

    # Loop through each image in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust extensions as needed
            image_path = os.path.join(folder_path, filename)

            # Read the image
            image = cv2.imread(image_path)
            if image is not None:
                # Resize the image
                image_resized = cv2.resize(image, new_size)

                # Normalize the image
                image_normalized = image_resized / 255.0

                # Save the preprocessed image
                cv2.imwrite(image_path, (image_normalized * 255).astype(np.uint8))  # Save it back as uint8

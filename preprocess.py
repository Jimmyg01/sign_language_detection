import os
import cv2
import numpy as np

# Set the path to your dataset
dataset_path = r'C:\\Users\\jimmy\\OneDrive\\Desktop\\sign_language_detection\\Data'

# Define the new image size (adjust this as needed)
new_size = (128, 128)  # This should be the same size you plan to use in your model

# Loop through each number folder
for number in range(10):
    folder_path = os.path.join(dataset_path, str(number))
    if not os.path.exists(folder_path):
        continue

    # Create a folder for preprocessed images (optional)
    preprocessed_folder = os.path.join(dataset_path, 'preprocessed', str(number))
    os.makedirs(preprocessed_folder, exist_ok=True)

    # Loop through each image in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust extensions as needed
            image_path = os.path.join(folder_path, filename)
            print(image_path)
            # Read the image
            image = cv2.imread(image_path)
            if image is not None:
                # Resize the image
                image_resized = cv2.resize(image, new_size)

                # Normalize the image
                image_normalized = image_resized / 255.0  # Normalize to [0, 1]

                # Save the preprocessed image in the new folder
                preprocessed_image_path = os.path.join(preprocessed_folder, filename)
                cv2.imwrite(preprocessed_image_path, (image_normalized * 255).astype(np.uint8))  # Save it back as uint8



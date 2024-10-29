import cv2
import os
import time


# Define the number you want to capture
label = '5'
num_images = 100  # Number of images to capture
save_path = f'../data/{label}/'

# Create the folder if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Open the webcam
cap = cv2.VideoCapture(0)
# Delay the program after the opening of the program to allow computer to load
time.sleep(2)

count = 0
while count < num_images:
    ret, frame = cap.read()
    if ret:
        # Display the frame
        cv2.imshow('frame', frame)
        time.sleep(0.1)

        # Save the image to the folder for data storage
        img_name = f'{save_path}/{label}_{count}.jpg'
        cv2.imwrite(img_name, frame)
        print(f'Saved: {img_name}')
        count += 1

        # Press 'q' to stop early if needed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

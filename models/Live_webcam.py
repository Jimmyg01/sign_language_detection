import cv2
import torch
import torchvision.transforms as transforms
from train_model import SignLanguageModel  # Import the model from train_model.py
import numpy as np
from PIL import Image

# Load your trained model
model = SignLanguageModel()
model.load_state_dict(torch.load('sign_language_model.pth'))
model.eval()  # Set the model to evaluation mode

# Define the transform to resize and normalize the input
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert numpy array to PIL Image
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),  # Ensure input is single channel
    transforms.ToTensor(),
])

def predict(image):
    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    # print(image_tensor)

    with torch.no_grad():  # No need to track gradients
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)  # Convert logits to probabilities
        confidence, predicted = torch.max(probabilities, 1)  # Get predicted class and confidence score

    return predicted.item(), confidence.item()  # Return both predicted class and confidence score

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Make a prediction on the frame
    predicted_number, confidence = predict(frame)

    # Overlay the prediction and confidence on the frame
    cv2.putText(frame, f'Prediction: {predicted_number}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Create a visual representation of confidence
    bar_width = 300  # Width of the confidence bar
    bar_height = 30  # Height of the confidence bar
    cv2.rectangle(frame, (10, 50), (10 + bar_width, 50 + bar_height), (0, 0, 0), -1)  # Background
    cv2.rectangle(frame, (10, 50), (10 + int(bar_width * confidence), 50 + bar_height), (0, 255, 0),-1)  # Confidence bar

    # Display the frame
    cv2.imshow('Sign Language Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

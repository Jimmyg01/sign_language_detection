import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class SignLanguageDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.images = []
        self.labels = []

        # Loop through each numbered folder
        for label in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, label)
            if os.path.isdir(folder_path):
                for img_file in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_file)
                    if img_file.endswith('.png') or img_file.endswith('.jpg'):  # Adjust based on the image formats
                        self.images.append(img_path)
                        self.labels.append(int(label))

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('L')  # Convert to grayscale if needed
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Define the data augmentation transformations used to vary the input data
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 256x256
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(15),  # Randomly rotate images by 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, etc.
    transforms.ToTensor(),  # Convert image to Tensor
    transforms.Normalize((0.5,), (0.5,)),  # Normalize the image
])

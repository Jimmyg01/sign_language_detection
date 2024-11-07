import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torchvision import transforms  # Import torchvision for data augmentation
from PIL import Image


# Define the SignLanguageDataset class
class SignLanguageDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images  # List of image file paths
        self.labels = labels  # Corresponding labels for the images
        self.transform = transform  # Optional transforms

    def __len__(self):
        return len(self.images)  # Return the number of samples

    def __getitem__(self, idx):
        # Load the image
        img_path = self.images[idx]
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        label = int(self.labels[idx])  # Convert label to integer if necessary

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label


# Define the CNN model
class SignLanguageModel(nn.Module):
    def __init__(self):
        super(SignLanguageModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 128x128 -> 128x128 (same size)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)        # 128x128 -> 64x64 (reduction in size)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 64x64 -> 64x64 (same size)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.5)

        # Using a dummy input to find the flattened size 
        dummy_input = torch.zeros(1, 1, 128, 128)  # Assuming your images are 128x128
        self.flattened_size = self._get_flattened_size(dummy_input)

        # Now set up the fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 64) #
        self.fc2 = nn.Linear(64, 10)  # Assuming you have 10 per classes

    def _get_flattened_size(self, x):
        x = self.conv1(x)    # Conv1: 128x128 -> 128x128
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.pool(x)     # Pool1: 128x128 -> 64x64
        x = self.conv2(x)    # Conv2: 64x64 -> 64x64
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)     # Pool2: 64x64 -> 32x32
        return x.view(x.size(0), -1).shape[1]  # Flatten and get size

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)  # Apply pooling to the second convolution output
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


# Set the path to your dataset
dataset_path = r'C:\\Users\\jimmy\\OneDrive\\Desktop\\sign_language_detection\\Data'

# Create lists to hold image paths and labels
all_images = []
all_labels = []

# Loop through each folder in the dataset directory
for label in os.listdir(dataset_path):
    label_folder = os.path.join(dataset_path, label)
    if os.path.isdir(label_folder):  # Check if it's a directory
        for img_file in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_file)
            if img_file.endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image file types
                all_images.append(img_path)
                all_labels.append(label)  # Use folder name as label

# Data augmentation transforms
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

# Split the dataset into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(all_images, all_labels, test_size=0.2,
                                                                      random_state=42)

# Create the dataset and data loaders
train_dataset = SignLanguageDataset(images=train_images, labels=train_labels, transform=data_transforms)
val_dataset = SignLanguageDataset(images=val_images, labels=val_labels, transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

# Start the model, loss function, and optimiser
model = SignLanguageModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Define training parameters
num_epochs = 7  # Adjust based on your requirements

Run through the epochs
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        running_loss += loss.item()

    # Validation phase
    val_running_loss = 0.0
    all_preds = []
    all_labels = []

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for validation
        for val_images, val_labels in val_loader:
            val_outputs = model(val_images)  # Forward pass on validation set
            val_loss = criterion(val_outputs, val_labels)  # Compute validation loss

            val_running_loss += val_loss.item()

            # Get predicted classes
            _, predicted = torch.max(val_outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(val_labels.numpy())

    # Calculate the average validation loss and accuracy
    val_loss_average = val_running_loss / len(val_loader)
    val_accuracy = accuracy_score(all_labels, all_preds)

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss_average:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%')

# Save the trained model
torch.save(model.state_dict(), 'sign_language_model.pth')
print("Model saved as sign_language_model.pth")

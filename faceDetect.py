import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
# Define the neural network
class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model, loss function, and optimizer
model = GenderClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
import matplotlib.pyplot as plt

# Load the model
model = GenderClassifier()
model.load_state_dict(torch.load('gender_classifier.pth'))  # Ensure to save the model state_dict during training
model.to(device)
model.eval()

# Define the transformations for the new image
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

import os

# Initialize an empty list to store the names of PNG files
images = []

# Loop through each file in the current directory
for file in os.listdir():
    # Check if the file ends with '.png'
    if file.endswith('.png'):
        # Print the name of the PNG file
        print(file)
        # Append the name to the list
        images.append(file)

image_path = input('Path to your image:')
image = Image.open(image_path)
image = transform(image)
image = image.unsqueeze(0)  # Add batch dimension

# Load the model
model = GenderClassifier()
model.load_state_dict(torch.load('gender_classifier.pth'))  # Ensure to save the model state_dict during training
model.to(device)
model.eval()

# Move the image to the device
image = image.to(device)

# Make the prediction
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    class_idx = predicted.item()

# Map the class index to class label
class_labels = ['Female', 'Male']
predicted_label = class_labels[class_idx]

print(f'The predicted gender is: {predicted_label}')


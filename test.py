import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
image_size = 64
save_model_path = "ev_charger_types_classifier.pth"  # Path to the saved model

# Define the model architecture
class EvChargerTypesClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(EvChargerTypesClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.AvgPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.AvgPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (image_size // 8) * (image_size // 8), 128),
            nn.Tanh(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load the model
classes = ['valid_charger', 'not_valid']  # Replace with your actual class names
# num_classes = 4
model = EvChargerTypesClassifier(len(classes)).to(device)

# Function to test the model on a single image
def test_model(image_path):
    # Load the model weights
    model.load_state_dict(torch.load(save_model_path))
    model.eval()  # Set model to evaluation mode

    # Define the image transformation
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        outputs = model(image.to(device))
        _, predicted = torch.max(outputs, 1)
        print(f"The model predicts: {classes[predicted.item()]}")

# Example usage
if __name__ == "__main__":
    # Path to the image for testing
    # Valid
    test_image_path0 = "Images/Test/img0.jpg"  # Replace with the actual path
    test_image_path1 = "Images/Test/img1.jpg"
    test_image_path2 = "Images/Test/charger_9-390-_png_jpg.rf.69ed6af6a0e57aadd9fcbe90706bf7b4.jpg"
    test_image_path4 = "Images/Test/img.png"
    # Not valid
    test_image_path3 = "Images/Test/IMG_3927.jpeg"

    # Test the model on the images
    test_model(test_image_path0)
    test_model(test_image_path1)
    test_model(test_image_path2)
    test_model(test_image_path3)
    test_model(test_image_path4)

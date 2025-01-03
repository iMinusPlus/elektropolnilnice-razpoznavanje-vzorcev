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
            nn.Linear(128, 1),  # Single output for binary classification
            nn.Sigmoid()  # Sigmoid for binary classification
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load the model
classes = ['charging_plug', 'other']  # Replace with your actual class names
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
        prediction = (outputs > 0.5).item()  # Threshold for binary classification
        print(f"The model predicts: {classes[prediction]}")

# Example usage
if __name__ == "__main__":
    # Path to the image for testing
    valid_image_path = [
        "test/12.jpg",
        "test/img.png",
        "dataset/charging_plug/1.jpg",
        "dataset/charging_plug/2.jpg",
        "dataset/charging_plug/3.jpg",
        "dataset/charging_plug/4.jpg",
        "dataset/charging_plug/5.jpg",
        "dataset/charging_plug/6.jpg",
        "dataset/charging_plug/7.jpg",
        "dataset/charging_plug/8.jpg",
        "dataset/charging_plug/9.jpg",
        "dataset/charging_plug/10.jpg",
        "dataset/charging_plug/11.jpg",
        "dataset/charging_plug/12.jpg",
        "dataset/charging_plug/13.jpg",
        "dataset/charging_plug/14.jpg",
        "dataset/charging_plug/15.jpg",
        "dataset/charging_plug/16.jpg"
    ]

    invalid_image_path = [
        "test/10.jpg",
        "dataset/other/1.jpg",
        "dataset/other/2.jpg",
        "dataset/other/3.jpg",
        "dataset/other/4.jpg",
        "dataset/other/5.jpg",
        "dataset/other/6.jpg",
        "dataset/other/7.jpg",
        "dataset/other/8.jpg",
        "dataset/other/9.jpg",
        "dataset/other/10.jpg",
        "dataset/other/11.jpg",
        "dataset/other/12.jpg",
        "dataset/other/13.jpg",
        "dataset/other/14.jpg",
        "dataset/other/15.jpg",
        "dataset/other/16.jpg"
    ]
    # Valid
    # test_image_path0 = "test/12.jpg"
    # test_image_path1 = "test/img.png"
    # # Not valid
    # test_image_path0 = "test/10.jpg"

    # Test the model on the images
    # test_model(test_image_path0)
    # test_model(test_image_path1)

    print("Valid images:")
    for image_path in valid_image_path:
        test_model(image_path)

    print("\nInvalid images:")
    for image_path in invalid_image_path:
        test_model(image_path)
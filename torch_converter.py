import torch
import torch.nn as nn
# from model_definition import EvChargerTypesClassifier  # Import your model class

image_size = 64

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

# Define the model
model = EvChargerTypesClassifier()

# Load the state dictionary
model.load_state_dict(torch.load('ev_charger_types_classifier.pth'))
model.eval()  # Set the model to evaluation mode

# Trace the model with a dummy input
example_input = torch.randn(1, 3, 64, 64)  # Adjust input size as per your model
traced_model = torch.jit.trace(model, example_input)

# Save the model as a TorchScript file
traced_model.save('ev_charger_types_classifier.pt')

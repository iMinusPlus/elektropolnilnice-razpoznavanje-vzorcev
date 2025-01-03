import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from tqdm import tqdm

# Preveri, ali je na voljo GPU; če ne, uporabi CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparametri
image_size = 64
batch_size = 32
num_epochs = 100
learning_rate = 0.001
data_dir = "dataset"  # Pot do podatkovnega seta
save_model_path = "ev_charger_types_classifier_2.pth"

# Transformacije za pripravo podatkov
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Nalaganje in razdelitev podatkov
dataset = datasets.ImageFolder(data_dir, transform=transform)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Arhitektura nevronske mreže
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
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Inicializacija modela, izgube in optimizacije
num_classes = len(dataset.classes)
model = EvChargerTypesClassifier(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # Zmanjšaj hitrost učenja na vsakih 30 epoh

# Funkcija za treniranje in validacijo
def train_and_validate():
    train_losses, val_losses = [], []
    val_accuracies = []

    for epoch in tqdm(range(num_epochs), desc="Training and Validating", unit="epoch"):
        model.train()
        train_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0
        val_correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accuracy = val_correct / total
        val_accuracies.append(val_accuracy)
        scheduler.step()

        print(f"\rEpoh {epoch + 1}/{num_epochs}, Izguba pri treniranju: {train_losses[-1]:.4f}, "
              f"Izguba pri validaciji: {val_losses[-1]:.4f}, Natančnost: {val_accuracy:.4f}")

    return train_losses, val_losses, val_accuracies

# Prikaz grafov izgube in natančnosti
def plot_metrics(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Izguba pri treniranju')
    plt.plot(val_losses, label='Izguba pri validaciji')
    plt.xlabel('Epohe')
    plt.ylabel('Izguba')
    plt.legend()
    plt.title('Izguba pri treniranju in validaciji')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validacijska natančnost')
    plt.xlabel('Epohe')
    plt.ylabel('Natančnost')
    plt.legend()
    plt.title('Validacijska natančnost')
    plt.show()

# Prikaz napovedi na validacijskem sklopu
def show_predictions(data_loader, num_images=5):
    model.eval()
    fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
    for i, (images, labels) in enumerate(data_loader):
        if i >= num_images:
            break
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        axs[i].imshow(transforms.ToPILImage()(images[0].cpu()))
        axs[i].set_title(f"Napoved: {dataset.classes[predicted[0].item()]}")
        axs[i].axis('off')
    plt.show()

# Shrani in naloži model
def save_model():
    torch.save(model.state_dict(), save_model_path)
    print(f"Model shranjen v {save_model_path}")

def load_model():
    model.load_state_dict(torch.load(save_model_path))
    model.eval()
    print("Model naložen za testiranje")

# Glavni izvedbeni blok
if __name__ == "__main__":
    train_losses, val_losses, val_accuracies = train_and_validate()
    plot_metrics(train_losses, val_losses, val_accuracies)
    save_model()
    show_predictions(val_loader)

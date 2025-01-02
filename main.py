import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm

# če je možno uporablja GPU če ne pa samo cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 64
batch_size = 32
num_epochs = 10  # število prehodov skozi celotni učni nabor
learning_rate = 0.001

data_dir = "Images/Training"
save_model_path = "ev_charger_types_classifier.pth"

# Data transformations: Preprocessing steps for the dataset
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # Vse slike nastavimo na isto velikost
    transforms.RandomHorizontalFlip(),  # naredimo naključni flip za augmentacijo
    transforms.RandomRotation(15),  # naredimo naključno rotacijo za augmentacijo
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # spremenimo barvo
    transforms.ToTensor(),  # Spremenimo sliko v pytorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normaliziramo tensor
])

# Naložimo in razdelimo dataset v training in validation del seta
dataset = datasets.ImageFolder(data_dir, transform=transform)  # Naložimo sliko z transformacijo
train_size = int(0.9 * len(dataset))  # 90% uporabimo za treniranje
val_size = len(dataset) - train_size  # 10% za validacijo
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Arhitektura nevronske mreže
class EvChargerTypesClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(EvChargerTypesClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # Prvi sloj konvolucije
            nn.ReLU(),  # Funkcija za aktivacijo(vse negativne vrednosti nadomesti z ničlo)
            nn.Dropout(0.25),  # regularizacija
            nn.AvgPool2d(2),  # izvede pooling

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Drugi sloj(16 ker je izhod prejšnjega sloja)
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Tretji sloj (32 spet izhod)
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.AvgPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Spremeni tenzor v enodimenzionalni array
            nn.Linear(64 * (image_size // 8) * (image_size // 8), 128),  # povezani linearni sloj
            nn.Tanh(),  # Aktvacijska funkcija (uvede nelinearnost)
            nn.Linear(128, num_classes),  # Povezan linearni sloj za binarno klasifikacijo
            nn.Softmax(dim=1)  # Pretvori surove vrednosti v verjetnosti
        )

    def forward(self, x):
        x = self.conv_layers(x)  # Konvolucijski sloji
        x = self.fc_layers(x)  # Povezani linearni sloji
        return x


# Inicializacija modela izgube in optimizatcije
num_classes = 4
model = EvChargerTypesClassifier(num_classes).to(device)  # Premakni model na ustrezno napravo (GPU ali CPU)
criterion = nn.CrossEntropyLoss()  # Funkcija izgube za klasifikacijo
optimizer = Adam(model.parameters(), lr=learning_rate)  # Adam optimizator


# Zanka za treniranje
def train_and_validate():
    train_losses, val_losses = [], []

    for epoch in tqdm(range(num_epochs), desc="Training and Validating", unit="epoch"):
        model.train()  # Nastavi model v način treniranja
        train_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Premakni podatke na ustrezno napravo

            # Propagacija naprej
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()  # Ponastavi gradient
            loss.backward()  # Povratno razširjanje (backpropagation)
            optimizer.step()  # Posodobi uteži modela

            train_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))  # Zabeleži izgubo pri treniranju

    # Validacija
    model.eval()  # Nastavi model v način validacije
    val_loss = 0
    with torch.no_grad():  # Onemogoči izračun gradientov
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_losses.append(val_loss / len(val_loader))  # Zabeleži izgubo pri validaciji

    print(
        f"Epoh [{epoch + 1}/{num_epochs}], Izguba pri treniranju: {train_losses[-1]:.4f}, Izguba pri validaciji: {val_losses[-1]:.4f}")

    return train_losses, val_losses


# Prikaz grafov izgube pri treniranju in validaciji
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Izguba pri treniranju')
    plt.plot(val_losses, label='Izguba pri validaciji')
    plt.xlabel('Epohe')
    plt.ylabel('Izguba')
    plt.legend()
    plt.title('Izguba pri treniranju in validaciji')
    plt.show()


# Shrani natreniran model v datoteko
def save_model():
    torch.save(model.state_dict(), save_model_path)
    print(f"Model shranjen v {save_model_path}")


# Naloži shranjen model za testiranje
def load_model():
    model.load_state_dict(torch.load(save_model_path))  # Naloži stanje modela iz datoteke
    model.eval()  # Nastavi model v način validacije/testiranja
    print("Model naložen za testiranje")


# Glavni izvedbeni blok
if __name__ == "__main__":
    train_losses, val_losses = train_and_validate()  # Trenira in validira model
    plot_loss(train_losses, val_losses)  # Prikaže grafe izgube
    save_model()  # Shrani natreniran model

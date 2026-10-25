
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# 1. Select device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# 2. Data transformations
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

eval_transform = transforms.ToTensor()


# 3. Create separate datasets
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=train_transform
)

eval_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=eval_transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=eval_transform
)


# 4. Reproducible train/validation split
train_size = 45000
val_size = 5000

generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    len(train_dataset),
    generator=generator
).tolist()

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_subset = Subset(train_dataset, train_indices)
validation_subset = Subset(eval_dataset, val_indices)


# 5. DataLoaders
train_loader = DataLoader(
    train_subset,
    batch_size=64,
    shuffle=True
)

validation_loader = DataLoader(
    validation_subset,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# 6. Define CNN
class DeeperCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x


# 7. Initialize model, loss and optimizer
model = DeeperCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.03,
    momentum=0.9,
    weight_decay=1e-4
)


# 8. Training with best-model checkpointing
epochs = 50
best_val_loss = float("inf")
checkpoint_path = "best_model.pth"

for epoch in range(epochs):

    # Training phase
    model.train()
    training_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        scores = model(images)
        loss = criterion(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

    avg_training_loss = training_loss / len(train_loader)

    # Validation phase
    model.eval()
    validation_loss = 0.0
    epoch_correct = 0
    epoch_total = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            scores = model(images)
            loss = criterion(scores, labels)

            validation_loss += loss.item()

            predictions = scores.argmax(dim=1)
            epoch_correct += (predictions == labels).sum().item()
            epoch_total += labels.size(0)

    avg_validation_loss = validation_loss / len(validation_loader)
    validation_accuracy = 100.0 * epoch_correct / epoch_total

    # Save checkpoint whenever validation loss improves
    if avg_validation_loss < best_val_loss:
        best_val_loss = avg_validation_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss
            },
            checkpoint_path
        )

        print(
            f"Best model saved at epoch {epoch + 1}, "
            f"Val Loss: {best_val_loss:.4f}"
        )

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Training Loss: {avg_training_loss:.4f}, "
        f"Val Loss: {avg_validation_loss:.4f}, "
        f"Val Accuracy: {validation_accuracy:.2f}%"
    )


# 9. Load the best checkpoint
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"\nBest checkpoint epoch: {checkpoint['epoch']}")
print(f"Best validation loss: {checkpoint['val_loss']:.4f}")


# 10. Final test evaluation
total = 0
correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        scores = model(images)
        predictions = scores.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

test_accuracy = 100.0 * correct / total

print(f"Test accuracy: {test_accuracy:.2f}%")

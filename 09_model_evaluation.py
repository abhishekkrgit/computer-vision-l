import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor 


# setting the device type
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.backends.cuda.is_available():
    device = torch.device("cuda")
else :
    device = torch.device("cpu")
print(f"using device:{device}" )


train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=ToTensor()
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=ToTensor()
)
train_size = 45000
val_size = 5000
generator = torch.Generator().manual_seed(42)

train_subset, validation_subset = random_split(
    train_dataset,
    [train_size, val_size],
    generator = generator
)

train_loader = DataLoader(
    train_subset,
    batch_size=64,
    shuffle=True
)

validation_loader = DataLoader(
    validation_subset,
    batch_size = 64,
    shuffle = False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle = False
)


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

        self.classifier = nn.Linear(64*4*4, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim = 1)
        x = self.classifier(x)
        return x


model = DeeperCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.03,
    momentum = 0.9
)

epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        scores = model(images)
        loss = criterion(scores, labels)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    epoch_accuracy = 0.0
    with torch.no_grad():
        epoch_correct = 0
        epoch_total = 0
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            scores = model(images)
            prediction = scores.argmax(dim=1)
            epoch_correct += (labels == prediction).sum().item()
            epoch_total += len(labels)

        epoch_accuracy = 100* epoch_correct/epoch_total

    avg_loss = running_loss/len(train_loader)
    print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {avg_loss:.4f}, "
            f"Accuracy: {epoch_accuracy:.4f}"
        )




model.eval()
total=0.0
correct=0.0

with torch.no_grad():

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        scores = model(images)
        prediction = scores.argmax(dim = 1)
        # print(f"dim of scores: {scores.shape}")
        # print(f"dim of scores: {prediction.shape}")

        total += len(prediction)
        correct += (prediction == labels).sum().item()
       

        



accuracy = 100*correct/total

print(f"accuracy is :{accuracy:.4f}")








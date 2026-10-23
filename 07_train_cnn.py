import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor 

#1:  selecting device
if torch.backends.mps.is_available(): # for apple processors
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else :
    device = torch.device("cpu")
print("Using device:", device)


train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=ToTensor()
)

test_dataset= datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=ToTensor()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Linear(16*16*16, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x




model = SimpleCNN().to(device)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.01,
    momentum=0.9
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

        # back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


    avg_loss = running_loss/len(train_loader)
    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {avg_loss:.4f}"
    )





# Evaluating on unseen test images

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        scores = model(images)
        predictions = scores.argmax(dim = 1)
        total += labels.size(0)
        correct +=(predictions == labels).sum().item()


accuracy = 100* correct/total

print(f"test accuracy : {accuracy:.2f}%")






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


#2:  loading cifar-10
train_data = datasets.CIFAR10(
    root = "./data",
    train = True,
    download = False,
    transform = ToTensor()
)

test_data = datasets.CIFAR10(
    root = "./data",
    train = False,
    download = False,
    transform = ToTensor()
)

train_loader = DataLoader(
    train_data,
    batch_size = 64,
    shuffle = True
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)

# 3:  Defining models
class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3072, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim = 1)
        return self.linear(x)


model = LinearClassifier().to(2)


#4: Defining loss and otimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01
)



#5: training model

epochs  = 500
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)  # forward pass
        loss = criterion(logits, labels)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {average_loss:.4f}"
    )



# 6. Evaluate on unseen test images
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total

print(f"Test accuracy: {accuracy:.2f}%")


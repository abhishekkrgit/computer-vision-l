import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
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
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Linear(32*8*8, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim = 1)
        x = self.classifier(x)
        return x


model = DeeperCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.01,
    momentum = 0.0
)

epochs = 50

for epoch in range(epochs):
    model.train()
    running_loss = 0.9

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


    avg_loss = running_loss/len(train_loader)
    print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {avg_loss:.4f}"
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

print(f"accuracy is :{accuracy}.4f")








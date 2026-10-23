import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor


dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=ToTensor()
)

image, label = dataset[0]
image = image.unsqueeze(0) # 3, 32, 32  => 1, 3, 32, 32

# CNN
conv = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    padding=1
)


# relu

relu = nn.ReLU()
pool = nn.MaxPool2d(kernel_size=2, stride=2)


# forward pass
x = conv(image)
print("After convolution: ", x.shape)

x = relu(x)
print("After relu: ", x.shape)

x = pool(x)
print("After max pooling: ", x.shape)

x = torch.flatten(x, start_dim=1)
print("After flatten:", x.shape)


classifier = nn.Linear(16*16*16, 10)

scores = classifier(x)

print("scores shape: ", scores.shape)
print("actual class: ", dataset.classes[label])
print("predicted class: ", dataset.classes[scores.argmax(dim=1).item()])





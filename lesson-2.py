
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Load CIFAR-10
dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,  #putting true will download it
    transform=ToTensor()
)

# Get one image and its label
image, label = dataset[0]

print("Image shape:", image.shape)
print("Data type:", image.dtype)
print("Minimum pixel value:", image.min().item())
print("Maximum pixel value:", image.max().item())
print("Label:", dataset.classes[label])

# Inspect one pixel's RGB values
pixel = image[:, 10, 15]

print("Pixel RGB values:", pixel)
print("Red:", pixel[0].item())
print("Green:", pixel[1].item())
print("Blue:", pixel[2].item())

# Display the image
plt.imshow(image.permute(1, 2, 0))
plt.title(dataset.classes[label])
plt.axis("off")
plt.show()

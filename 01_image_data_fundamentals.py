
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# Load CIFAR-10
dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,  #putting true will download it
    transform=ToTensor()
)

# Get one image and its label
image, label = dataset[0]
print ("single elem of dataset: ", dataset[0])
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

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

images, labels = next(iter(loader))

print("Batch image shape:", images.shape)
print("Batch labels shape:", labels.shape)
print("First 10 labels:", labels[:10]) # correct label of image e.g. First 10 labels: tensor([5, 7, 6, 2, 6, 6, 1, 0, 2, 0])



# Display the image
# plt.imshow(image.permute(1, 2, 0))
# plt.title(dataset.classes[label])
# plt.axis("off")
# plt.show()
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor


dataset = datasets.CIFAR10(
    root = "./data",
    train = True,
    download = False,
    transform = ToTensor()
)

image, label = dataset[0]

image = image.unsqueeze(0) # Add batch dimension: (3, 32, 32) -> (1, 3, 32, 32)

conv = nn.Conv2d(
    in_channels = 3,
    out_channels = 16,
    kernel_size = 3,
    stride = 1,
    padding = 1
)

output = conv(image)

print("image shape: ", image.shape)
print("output shape: ", output.shape)
print("Filter weights:", conv.weight.shape)
print("Class:", dataset.classes[label])
print("Trainable parameters:", sum(
    p.numel() for p in conv.parameters() if p.requires_grad
))




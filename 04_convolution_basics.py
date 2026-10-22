import torch
import torch.nn as nn


# one gray scale image: channel = 1, h = 5, w = 5, batch = 1

image = torch.tensor(
[
    [0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 0.],
    [0., 1., 1., 1., 0.],
    [0., 1., 1., 1., 0.],
    [0., 0., 0., 0., 0.]
])

image = image.unsqueeze(0).unsqueeze(0)

conv = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    padding=1,
    bias=False
)

kernel = torch.tensor([
    [1., 0., -1.],
    [1., 0., -1.],
    [1., 0., -1.]
])

with torch.no_grad():
    conv.weight.copy_(kernel.reshape(1, 1, 3, 3))


output = conv(image)

print("Input shape:", image.shape)
print("Filter shape: ", conv.weight.shape)
print("Output shape: ", output.shape)
print("feature map: \n", output[0, 0])

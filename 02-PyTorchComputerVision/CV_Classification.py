import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

print(f"torchvision version: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

image, label = train_data[0]

# print(train_data[0])
print(1)
print(image.shape)
plt.imshow(image.squeeze())
plt.title(label)
plt.show()
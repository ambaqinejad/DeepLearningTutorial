import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np

torch.manual_seed(7)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 1

# Load training dataset into a single batch to cumpute mean and stddev

transform = transforms.Compose([transforms.ToTensor()])
trainset = MNIST(root="./dataset", train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=True)
data = next(iter(trainloader))
mean = data[0].mean()
stddev = data[0].std()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, stddev)
])

trainset = MNIST(root="./dataset", train=True, download=True, transform=transform)
testset = MNIST(root="./dataset", train=False, download=True, transform=transform)

# Create s sequential (feed-forward) model
# 784 input = 28 * 28 (image size)
# Two fully connected layers with 25 and 10 neurons
# tanh as activation function for hidden layer
# logistic (sigmoid) as activation function for output layer.

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=784, out_features=25),
    nn.Tanh(),
    nn.Linear(in_features=25, out_features=10),
    nn.Sigmoid()
)

for module in model.modules():
    if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, a=-.1, b=.1)
        nn.init.constant_(module.bias, 0.0)

print(1)


optimizer = torch.optim.SGD(model.parameters(), lr=.01)
loss_function = nn.MSELoss()

model.to(device)
trainloader = DataLoader(trainset, BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, BATCH_SIZE, shuffle=False)

for i in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_batches = 0
    for inputs, targets in trainloader:
        one_hot_targets = nn.functional.one_hot(targets, num_classes=10).float()
        inputs, targets, one_hot_targets = inputs.to(device), targets.to(device), one_hot_targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, one_hot_targets)

        _, indices = torch.max(outputs.data, 1)
        train_correct += (indices == targets).sum().item()
        train_batches += 1
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / train_batches
    train_acc = train_correct / (train_batches * BATCH_SIZE)

    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_batches = 0

    for inputs, targets in testloader:
        one_hot_targets = nn.functional.one_hot(targets, num_classes=10).float()
        inputs, targets, one_hot_targets = inputs.to(device), targets.to(device), one_hot_targets.to(device)
        outputs = model(inputs)
        loss = loss_function(outputs, one_hot_targets)

        _, indices = torch.max(outputs.data, 1)
        test_correct += (indices == targets).sum().item()
        test_batches += 1
        test_loss += loss.item()
    test_loss = test_loss / test_batches
    test_acc = test_correct / (test_batches * BATCH_SIZE)

    print(f'EPOCH {i+1}/{EPOCHS} loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {test_loss:.4f} - val_acc: {test_acc:.4f}')
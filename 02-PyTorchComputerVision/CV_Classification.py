import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from helper_functions import accuracy_fn
from timeit import default_timer as timer

print(f"torchvision version: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")

device = "cuda" if torch.cuda.is_available() else "cpu"

class Model1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, X):
        return self.layer_stack(X)

class Model2(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, X):
        return self.layer_stack(X)


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

torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows*cols + 1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze())
# plt.show()
BATCH_SIZE = 32
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"Dataloaders: {train_data_loader, test_data_loader}")
print(f"Length of train dataloader: {len(train_data_loader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_data_loader)} batches of {BATCH_SIZE}")

model = Model2(28*28, 10, len(train_data.classes))
model.to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.1)

from timeit import default_timer as timer
def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

from tqdm.auto import tqdm

torch.manual_seed(42)
train_timer_start_on_gpu = timer()

epochs = 30

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_loss = 0
    for batch, (X, y) in enumerate(train_data_loader):
        model.train()
        X = X.to(device)
        y = y.to(device)
        # Forward Pass
        y_pred = model(X)
        # Calculate Loss(per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        # Optimizer zero grad
        optimizer.zero_grad()
        # loss backward
        loss.backward()
        # Optimizer Step
        optimizer.step()
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)} / {len(train_data_loader.dataset)} samples")

    train_loss /= len(train_data_loader)

    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in test_data_loader:
            X = X.to(device)
            y = y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y, test_pred.argmax(dim=1))
        test_loss /= len(test_data_loader)
        test_acc /= len(test_data_loader)
    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

train_timer_end_on_gpu = timer()
total_train_time_model = print_train_time(start=train_timer_start_on_gpu, end=train_timer_end_on_gpu, device=device)

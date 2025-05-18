import torch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, X):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(X)))))


def create_dataset(n_samples = 1000):
    return make_circles(n_samples=n_samples, noise=.03, random_state=42)


def plot_dataset(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu)
    plt.show()

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc


device = "cuda" if torch.cuda.is_available() else "cpu"
X, y = create_dataset()
plot_dataset(X, y)
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
model = Model().to(device)
X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.1)

torch.manual_seed(42)
epochs = 10000

for epoch in range(epochs):
    model.train()
    y = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y))
    loss = loss_fn(y, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        test = model(X_test).squeeze()
        test_loss = loss_fn(test, y_test)
        test_pred = torch.round(torch.sigmoid(test)).squeeze()
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

        # Print out what's happening
        if epoch % 100 == 0:
            print(
                f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {test_acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train) # model_1 = no non-linearity
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test) # model_3 = has non-linearity
plt.show()
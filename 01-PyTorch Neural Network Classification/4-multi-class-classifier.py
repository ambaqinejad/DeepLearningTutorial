import torch
from torch import nn
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary


def create_dataset(n_samples=1000, n_features=2, random_state=42, n_classes=4, cluster_std=1.5):
    return make_blobs(n_samples=n_samples, n_features=n_features, random_state=random_state, centers=n_classes, cluster_std=cluster_std)


def calc_acc(y_pred, y_true):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_true)) * 100
    return acc


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=20)
        self.layer3 = nn.Linear(in_features=20, out_features=4)
        self.relu = nn.ReLU()

    def forward(self, X):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(X)))))


device = "cuda" if torch.cuda.is_available() else "cpu"
X, y = create_dataset()
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.LongTensor)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)
torch.manual_seed(42)
loss_fn = nn.CrossEntropyLoss()
model = Model().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=.1)

epochs = 1000
for epoch in range(epochs):
    model.train()
    y_logits = model(X_train).squeeze()
    train_loss = loss_fn(y_logits, y_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    optimizer.zero_grad()
    train_loss.backward()
    acc = calc_acc(y_pred, y_train)
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits, y_test)
        test_acc = calc_acc(test_pred, y_test)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {train_loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
plt.show()
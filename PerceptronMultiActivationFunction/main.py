import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

# Generate Dataset
X, y = make_moons(n_samples=1000, noise=.2, random_state=42)
# X, y = make_circles(n_samples=1000, noise=.5, random_state=42)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, label="Train")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("Dataset")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# Model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = activation_fn
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        act_sig = nn.Sigmoid()
        return act_sig(self.layer2(self.activation(self.layer1(x))))


# Activation Function
activation_array = [nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Softmax, nn.Softmin, nn.Softsign, nn.Softplus]
activation_names = ['nn.ReLU', 'nn.Tanh', 'nn.Sigmoid', 'nn.Softmax', 'nn.Softmin', 'nn.Softsign', 'nn.Softplus']
loss_base_activation = []
for i in range(len(activation_array)):
    activation = activation_array[i]
    activation_name = activation_names[i]
    print(activation_name)

    activation_fn = activation()
    model = MLP(input_dim=2, hidden_dim=16, output_dim=2, activation_fn=activation_fn)
    print(model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=.01)

    n_epochs = 1000
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            pred = torch.argmax(output, dim=1)
            acc = (pred == y_train).float().mean()
            print(f"Epoch [{epoch + 1}/{n_epochs}] Loss: {loss.item():.4f} Acc: {acc:.4f}")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_preds = torch.argmax(test_outputs, dim=1)
        test_acc = (test_preds == y_test).float().mean()
        print(f"Test accuracy: {test_acc:.4f}")
        loss_base_activation.append(test_acc)
plt.figure(figsize=(8, 6))
plt.bar(activation_names, loss_base_activation, color='skyblue')

plt.title('Activation Function vs Accuracy')
plt.xlabel('Activation Function')
plt.ylabel('Accuracy')

for i, acc in enumerate(loss_base_activation):
    plt.text(i, acc + 0.01, f"{acc:.2f}", ha='center')

plt.ylim(0, 1.1)  # تا کمی بالاتر از 1 که فضا برای اعداد باز شود
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
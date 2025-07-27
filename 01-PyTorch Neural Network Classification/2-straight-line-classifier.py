import torch
from torch import nn
from helper_functions import plot_predictions


def create_dataset():
    slope = -10
    bias = 40
    start = 0
    end = 1
    step = .01
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = slope * X + bias
    return X, y


device = "cuda" if torch.cuda.is_available() else "cpu"
X, y = create_dataset()
train_split = int(.8 * len(X))
X_train, X_test = X[:train_split], X[train_split:]
y_train, y_test = y[:train_split], y[train_split:]

plot_predictions(train_data=X_train, test_data=X_test,
                 train_labels=y_train, test_labels=y_test)

X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

model = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
).to(device)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=.01)
torch.manual_seed(42)
epochs = 10000
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss:.5f}, Test loss: {test_loss:.5f}")
plot_predictions(train_data=X_train.cpu(),
                     train_labels=y_train.cpu(),
                     test_data=X_test.cpu(),
                     test_labels=y_test.cpu(),
                     predictions=test_pred.cpu());
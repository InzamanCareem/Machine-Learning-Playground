from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim


def get_data():
    data = fetch_california_housing()
    return data.data, data.target


def make_train_test(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def preprocess(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train, X_test, y_train, y_test


def make_model():
    class LinearRegressionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Sequential(
                nn.Linear(8, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            return self.layer(x)

    return LinearRegressionModel()


def get_loss_func(loss_name):
    if loss_name == "Mean Squared Error":
        return nn.MSELoss()
    elif loss_name == "Mean Absolute Error":
        return nn.L1Loss()
    elif loss_name == "Huber Loss":
        return nn.HuberLoss()
    else:
        raise ValueError("Unknown loss function")


def get_optimizer(opt_name, lr, model):
    if opt_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "SGD":
        return optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Unknown optimizer")


def model_train(model, loss_fn, optimizer, X_train, X_test, y_train, y_test, progress_callback=None):
    torch.manual_seed(42)

    epochs = 50

    epoch_count = []
    train_loss_values = []
    test_loss_values = []

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

        epoch_count.append(epoch)
        train_loss_values.append(loss.item())
        test_loss_values.append(test_loss.item())

        if progress_callback:
            progress_callback(int(((epoch + 1) / epochs) * 100))

    return epoch_count, train_loss_values, test_loss_values

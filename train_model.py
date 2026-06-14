import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim


def get_data(dataset_type, n_samples, n_features):
    if dataset_type == "Regression":
        data = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_features, n_targets=1,
                               shuffle=True, random_state=42)
        return data[0], data[1]

    elif dataset_type == "Classification":
        data = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features, n_redundant=0,
                                   n_repeated=0, n_classes=2, shuffle=True, random_state=42)
        return data[0], data[1]

    # TODO: Add many datasets


def make_train_test(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def preprocess(X_train, X_test, y_train, y_test):
    # TODO: Add many scalers
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train, X_test, y_train, y_test


def make_model(dataset, model, features):
    if dataset == "Regression":
        if model == "LinearRegression":
            return LinearRegression()
        elif model == "DecisionTreeRegressor":
            return DecisionTreeRegressor()
        elif model == "RandomForestRegressor":
            return RandomForestRegressor()
        elif model == "SVR":
            return SVR()
        elif model == "KNeighborsRegressor":
            return KNeighborsRegressor()
        elif model == "Custom Neural Network":
            class LinearRegressionModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layer = nn.Sequential(
                        nn.Linear(features, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Linear(16, 1)
                    )

                def forward(self, x):
                    return self.layer(x)

            return LinearRegressionModel()

    elif dataset == "Classification":
        if model == "LogisticRegression":
            return LogisticRegression()
        elif model == "DecisionTreeClassifier":
            return DecisionTreeClassifier()
        elif model == "RandomForestClassifier":
            return RandomForestClassifier()
        elif model == "SVC":
            return SVC()
        elif model == "KNeighborsClassifier":
            return KNeighborsClassifier()
        elif model == "Custom Neural Network":
            class BinaryClassificationModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layer = nn.Sequential(
                        nn.Linear(features, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Linear(16, 1),
                    )

                def forward(self, x):
                    return self.layer(x)

            return BinaryClassificationModel()


def make_learning_curve(model, X, y, scoring):
    # TODO: add ComboBox for scoring
    return learning_curve(model, X, y, cv=5, scoring=scoring, train_sizes=np.linspace(0.1, 1, 10))


def make_validation_curve(model, X, y, param_name, param_range, scoring):
    return validation_curve(model, X, y, param_name=param_name, param_range=param_range, cv=5, scoring=scoring)


def get_loss_func(loss_name):
    if loss_name == "Mean Squared Error":
        return nn.MSELoss()
    elif loss_name == "Mean Absolute Error":
        return nn.L1Loss()
    elif loss_name == "Huber Loss":
        return nn.HuberLoss()
    elif loss_name == "Binary Cross Entropy":
        return nn.BCEWithLogitsLoss()
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


def model_train(dataset_type, model, loss_fn, optimizer, X_train, X_test, y_train, y_test,
                progress_callback=None):
    torch.manual_seed(42)
    epochs = 50
    epoch_count = []
    train_loss_values = []
    test_loss_values = []
    train_accuracy_values = []
    test_accuracy_values = []

    if dataset_type == "Regression":
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

    elif dataset_type == "Classification":
        for epoch in range(epochs):
            model.train()

            y_logits = model(X_train)
            y_pred = torch.round(torch.sigmoid(y_logits))

            loss = loss_fn(y_logits, y_train)
            train_accuracy = accuracy_score(y_train.detach().numpy(), y_pred.detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_count.append(epoch)
            train_loss_values.append(loss.item())
            train_accuracy_values.append(train_accuracy)

            model.eval()
            with torch.inference_mode():
                test_logits = model(X_test)
                test_pred = torch.round(torch.sigmoid(test_logits))

                test_loss = loss_fn(test_logits, y_test)
                test_accuracy = accuracy_score(y_test.detach().numpy(), test_pred.detach().numpy())

                test_loss_values.append(test_loss.item())
                test_accuracy_values.append(test_accuracy)

            if progress_callback:
                progress_callback(int(((epoch + 1) / epochs) * 100))

        return epoch_count, train_loss_values, test_loss_values, train_accuracy_values, test_accuracy_values

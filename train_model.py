import numpy as np
import torch
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, \
    accuracy_score
from torch import nn, optim


def get_data(dataset_type, n_samples, n_features, noise):
    if dataset_type == "Regression":
        data = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_features, noise=noise,
                               n_targets=1, shuffle=True, random_state=42)
        return data[0], data[1]

    elif dataset_type == "Classification":
        data = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features, n_redundant=0,
                                   n_repeated=0, n_classes=2, shuffle=True, random_state=42)
        return data[0], data[1]


def make_train_test(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def preprocess(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_test, y_test


def make_model(model, features, **kwargs):
    if model == "Linear Regression":
        return LinearRegression(**kwargs)
    elif model == "Decision Tree Regressor":
        return DecisionTreeRegressor(**kwargs)
    elif model == "Random Forest Regressor":
        return RandomForestRegressor(**kwargs)
    elif model == "Support Vector Regressor":
        return SVR(**kwargs)
    elif model == "KNeighbors Regressor":
        return KNeighborsRegressor(**kwargs)
    elif model == "Custom Neural Network Regressor":
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

    elif model == "Logistic Regression":
        return LogisticRegression(**kwargs)
    elif model == "Decision Tree Classifier":
        return DecisionTreeClassifier(**kwargs)
    elif model == "Custom Neural Network Classifier":
        class BinaryClassificationModel(nn.Module):
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

        return BinaryClassificationModel()


def model_fit(model, X_train, y_train):
    return model.fit(X_train, y_train)


def model_predict(model, X_test):
    return torch.tensor(model.predict(X_test))


def model_predict_proba(model, X_test):
    return torch.tensor(model.predict_proba(X_test)[:, 1])


def get_confusion_matrix(y_true, y_prediction):
    return confusion_matrix(y_true, y_prediction)


def get_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def get_precision_recall_curve(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    return precision, recall


def get_average_precision_score(y_true, y_score):
    return average_precision_score(y_true, y_score)


def get_learning_curve(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(model, X, y, cv=5,
                                                           train_sizes=np.linspace(0.1, 1.0, 5))

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    return train_sizes, train_mean, val_mean


def get_validation_curve(model, X, y, param_name, param_range, scoring):
    if param_name is not None:
        train_scores, test_scores = validation_curve(model, X, y, param_name=param_name, param_range=param_range, cv=5,
                                                     scoring=scoring)

        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)

        return train_mean, test_mean

    return None, None


def get_loss_func(loss_name):
    if loss_name == "Mean Squared Error":
        return nn.MSELoss()
    elif loss_name == "Mean Absolute Error":
        return nn.L1Loss()
    elif loss_name == "Huber Loss":
        return nn.HuberLoss()
    elif loss_name == "Binary Cross Entropy":
        return nn.BCEWithLogitsLoss()


def get_optimizer(opt_name, lr, model):
    if opt_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "SGD":
        return optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=lr)


def train_regressor_model(model, loss_function_name, optimizer_name, lr, X_train, X_test, y_train, y_test):
    loss_fn = get_loss_func(loss_function_name)
    optimizer = get_optimizer(optimizer_name, lr, model)

    torch.manual_seed(42)
    epochs = 50
    epoch_count = []
    train_loss_values = []
    test_loss_values = []

    y_train = y_train.unsqueeze(1)
    y_test = y_test.unsqueeze(1)

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

    model.eval()
    with torch.inference_mode():
        predictions = model(X_test).squeeze()

    return epoch_count, train_loss_values, test_loss_values, predictions


def train_classifier_model(model, loss_function_name, optimizer_name, lr, X_train, X_test, y_train, y_test):
    loss_fn = get_loss_func(loss_function_name)
    optimizer = get_optimizer(optimizer_name, lr, model)

    torch.manual_seed(42)
    epochs = 50
    epoch_count = []
    train_loss_values = []
    test_loss_values = []
    train_accuracy_values = []
    test_accuracy_values = []

    y_train = y_train.unsqueeze(1)
    y_test = y_test.unsqueeze(1)

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

    model.eval()
    with torch.inference_mode():
        prediction_logits = model(X_test).squeeze()
        predictions = torch.round(torch.sigmoid(test_logits))

    return (epoch_count, train_loss_values, test_loss_values, train_accuracy_values, test_accuracy_values,
            prediction_logits, predictions)

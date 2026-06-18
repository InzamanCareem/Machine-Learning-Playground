import torch
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train, y_train, X_test, y_test


def make_model(model, **kwargs):
    if model == "Linear Regression":
        return LinearRegression(**kwargs)
    elif model == "Decision Tree Regressor":
        return DecisionTreeRegressor(**kwargs)


def model_fit(model, X_train, y_train):
    return model.fit(X_train, y_train)


def model_predict(model, X_test):
    return torch.tensor(model.predict(X_test))

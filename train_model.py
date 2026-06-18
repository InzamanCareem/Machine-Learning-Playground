import numpy as np
import torch
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score


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


def make_model(model, **kwargs):
    if model == "Linear Regression":
        return LinearRegression(**kwargs)
    elif model == "Decision Tree Regressor":
        return DecisionTreeRegressor(**kwargs)
    elif model == "Logistic Regression":
        return LogisticRegression(**kwargs)
    elif model == "Decision Tree Classifier":
        return DecisionTreeClassifier(**kwargs)


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

from PyQt6.QtCore import QThread, pyqtSignal

from train_model import *


class TrainWorker(QThread):
    run_config = pyqtSignal(dict)
    progress = pyqtSignal(int)

    def __init__(self, dataset, samples, features, model, lr=None, loss_name=None, optimizer_name=None):
        super().__init__()

        self.dataset = dataset
        self.samples = samples
        self.features = features
        self.model = model

        self.lr = lr
        self.loss_name = loss_name
        self.optimizer_name = optimizer_name

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    @classmethod
    def from_ml_model(cls, dataset, samples, features, model):
        return cls(dataset=dataset, samples=samples, features=features, model=model)

    @classmethod
    def from_dl_model(cls, dataset, samples, features, model, lr, loss_name, optimizer_name):
        return cls(dataset=dataset, samples=samples, features=features, model=model, lr=lr, loss_name=loss_name,
                   optimizer_name=optimizer_name)

    def _load_dataset(self):
        X, y = get_data(self.dataset, self.samples, self.features)
        self.X = X
        self.y = y
        X_train, X_test, y_train, y_test = make_train_test(X, y)
        X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def run(self):
        self._load_dataset()

        model = make_model(self.dataset, self.model, self.features)

        if self.dataset == "Regression":
            if self.model == "Custom Neural Network":
                loss_fn = get_loss_func(self.loss_name)
                opt = get_optimizer(self.optimizer_name, self.lr, model)

                epoch_count, train_loss, test_loss = model_train(self.dataset, model, loss_fn, opt, self.X_train,
                                                                 self.X_test, self.y_train, self.y_test,
                                                                 self.progress.emit)
                self.run_config.emit({
                    "epochs": epoch_count,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "model_type": "dlr",
                    "name": ""
                })

            else:
                train_sizes, train_scores, val_scores = make_learning_curve(model, self.X, self.y,
                                                                            "neg_root_mean_squared_error")

                self.run_config.emit({
                    "train_sizes": train_sizes,
                    "train_mean": train_scores.mean(axis=1),
                    "val_mean": val_scores.mean(axis=1),
                    "model_type": "ml",
                    "name": ""
                })

        if self.dataset == "Classification":
            if self.model == "Custom Neural Network":
                loss_fn = get_loss_func(self.loss_name)
                opt = get_optimizer(self.optimizer_name, self.lr, model)

                (epoch_count, train_loss, test_loss, train_accuracy,
                 test_accuracy) = model_train(self.dataset, model, loss_fn, opt, self.X_train, self.X_test,
                                              self.y_train,
                                              self.y_test, self.progress.emit)

                self.run_config.emit({
                    "epochs": epoch_count,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                    "model_type": "dlc",
                    "name": ""
                })

            else:
                train_sizes, train_scores, val_scores = make_learning_curve(model, self.X, self.y, "accuracy")

                self.run_config.emit({
                    "train_sizes": train_sizes,
                    "train_mean": train_scores.mean(axis=1),
                    "val_mean": val_scores.mean(axis=1),
                    "model_type": "ml",
                    "name": ""
                })

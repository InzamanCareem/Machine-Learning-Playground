from PyQt6.QtCore import QThread, pyqtSignal
from train_model import *


class TrainWorker(QThread):
    finished = pyqtSignal(object, object, object, object, object)
    progress = pyqtSignal(int)

    def __init__(self, dataset, samples, features, lr, loss_name, optimizer_name):
        super().__init__()
        self.dataset = dataset
        self.samples = samples
        self.features = features
        self.lr = lr
        self.loss_name = loss_name
        self.optimizer_name = optimizer_name
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_dataset(self):
        X, y = get_data(self.dataset, self.samples, self.features)
        X_train, X_test, y_train, y_test = make_train_test(X, y)
        X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def run(self):
        self.load_dataset()

        model = make_model(self.dataset, self.features)

        loss_fn = get_loss_func(self.loss_name)
        opt = get_optimizer(self.optimizer_name, self.lr, model)

        if self.dataset == "Regression":
            epoch_count, train_loss, test_loss = model_train(self.dataset, model, loss_fn, opt, self.X_train,
                                                             self.X_test, self.y_train, self.y_test, self.progress.emit)

            self.finished.emit(epoch_count, train_loss, test_loss, -1, -1)

        elif self.dataset == "Classification":
            (epoch_count, train_loss, test_loss, train_accuracy,
             test_accuracy) = model_train(self.dataset, model, loss_fn, opt, self.X_train, self.X_test, self.y_train,
                                          self.y_test, self.progress.emit)

            self.finished.emit(epoch_count, train_loss, test_loss, train_accuracy, test_accuracy)

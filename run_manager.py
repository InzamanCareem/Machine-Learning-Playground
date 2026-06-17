from PyQt6.QtCore import QThreadPool

from train_worker import TrainWorker
from train_model import get_data, make_train_test, preprocess, make_model


class RunManager:
    def __init__(self, plot_panel):
        self.plot_panel = plot_panel

        self.threadpool = QThreadPool()

        self.X = None
        self.y = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.model = None

    def load_dataset(self, dataset_type, samples, features):
        self.X, self.y = get_data(dataset_type, samples, features)

    def start(self):
        self.X_train, self.y_train, self.X_test, self.y_test = make_train_test(self.X, self.y)
        self.X_train, self.y_train, self.X_test, self.y_test = preprocess(self.X_train, self.y_train, self.X_test,
                                                                          self.y_test)

        self.model = make_model()

        print(self.X_train.shape)
        print(self.y_train.shape)
        print(self.X_test.shape)
        print(self.y_test.shape)

        train_worker = TrainWorker(self.X_train, self.y_train, self.X_test, self.y_test, self.model)

        train_worker.signals.run_config.connect(self.plot_panel.plot_curve)
        # train_worker.signals.run_config.connect(self.progress_panel.save_run)

        self.threadpool.start(train_worker)

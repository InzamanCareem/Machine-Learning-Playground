from PyQt6.QtCore import QThreadPool

from train_worker import TrainWorker
from train_model import get_data, make_train_test, preprocess, make_model


class RunManager:
    def __init__(self, dataset_type_controls, plot_panel, progress_panel):
        self.dataset_type_controls = dataset_type_controls
        self.plot_panel = plot_panel
        self.progress_panel = progress_panel

        self.threadpool = QThreadPool()

        self.X = None
        self.y = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.model = None

    def load_dataset(self, samples, features, noise):
        self.X, self.y = get_data(self.dataset_type_controls.get_dataset_type(), samples, features, noise)

    def load_model(self, model, **kwargs):
        self.model = make_model(model, **kwargs)

    def start(self):

        dataset_type = self.dataset_type_controls.get_dataset_type()

        print(dataset_type, "on run manager")

        self.X_train, self.y_train, self.X_test, self.y_test = make_train_test(self.X, self.y)
        self.X_train, self.y_train, self.X_test, self.y_test = preprocess(self.X_train, self.y_train, self.X_test,
                                                                          self.y_test)

        train_worker = TrainWorker(dataset_type, self.X_train, self.y_train, self.X_test, self.y_test, self.model)

        if dataset_type == "Regression":
            train_worker.signals.run_config.connect(self.plot_panel.plot_actual_vs_predicted)
            train_worker.signals.run_config.connect(self.plot_panel.plot_residuals_vs_fitted)

        elif dataset_type == "Classification":
            train_worker.signals.run_config.connect(self.plot_panel.plot_confusion_matrix)
            train_worker.signals.run_config.connect(self.plot_panel.plot_roc_curve)

        train_worker.signals.run_config.connect(self.progress_panel.save_run)

        self.threadpool.start(train_worker)

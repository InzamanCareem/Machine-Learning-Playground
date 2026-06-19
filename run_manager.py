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
        self.model_name = None

        self.param_name = None
        self.param_range = None
        self.scoring = None

        self.lr = None
        self.loss = None
        self.optimizer = None

        self.set_ui = None

    def set_model_controls_ui(self, set_ui):
        self.set_ui = set_ui

    def load_dataset(self, samples, features, noise):
        self.X, self.y = get_data(self.dataset_type_controls.get_dataset_type(), samples, features, noise)

    def load_model(self, model, **kwargs):
        self.model_name = model
        self.model = make_model(model, self.X.shape[1], **kwargs)

        self.lr = kwargs.get("lr", self.lr)
        self.loss = kwargs.get("loss", self.loss)
        self.optimizer = kwargs.get("optimizer", self.optimizer)

    def load_parameters(self, param_name, param_range, scoring):
        self.param_name = param_name
        self.param_range = param_range
        self.scoring = scoring

    def start(self):

        self.progress_panel.reset_progress_value()

        dataset_type = self.dataset_type_controls.get_dataset_type()

        self.X_train, self.y_train, self.X_test, self.y_test = make_train_test(self.X, self.y)
        self.X_train, self.y_train, self.X_test, self.y_test = preprocess(self.X_train, self.y_train, self.X_test,
                                                                          self.y_test)

        train_worker = TrainWorker(dataset_type, self.X, self.y, self.X_train, self.y_train, self.X_test, self.y_test,
                                   self.model, self.model_name, self.param_name, self.param_range, self.scoring,
                                   self.loss, self.optimizer, self.lr)

        if dataset_type == "Regression":
            train_worker.signals.run_config.connect(self.plot_panel.plot_actual_vs_predicted)
            train_worker.signals.run_config.connect(self.plot_panel.plot_residuals_vs_fitted)

            if self.model_name == "Custom Neural Network Regressor":
                train_worker.signals.run_config.connect(self.plot_panel.plot_loss_curve)
            else:
                train_worker.signals.run_config.connect(self.plot_panel.plot_learning_curve)

                if self.param_name is not None and self.param_range is not None and self.scoring is not None:
                    train_worker.signals.run_config.connect(self.plot_panel.plot_validation_curve)

        elif dataset_type == "Classification":
            train_worker.signals.run_config.connect(self.plot_panel.plot_confusion_matrix)
            train_worker.signals.run_config.connect(self.plot_panel.plot_roc_curve)
            train_worker.signals.run_config.connect(self.plot_panel.plot_precision_vs_recall)

            if self.model_name == "Custom Neural Network Classifier":
                train_worker.signals.run_config.connect(self.plot_panel.plot_loss_curve)
                train_worker.signals.run_config.connect(self.plot_panel.plot_accuracy_curve)
            else:
                train_worker.signals.run_config.connect(self.plot_panel.plot_learning_curve)

                if self.param_name is not None and self.param_range is not None and self.scoring is not None:
                    train_worker.signals.run_config.connect(self.plot_panel.plot_validation_curve)

        train_worker.signals.progress.connect(self.progress_panel.progress.setValue)
        train_worker.signals.run_config.connect(self.progress_panel.save_run)

        train_worker.signals.finished.connect(self.on_finished)

        self.threadpool.start(train_worker)

    def on_finished(self):
        self.set_ui(True)

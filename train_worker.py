from PyQt6.QtCore import pyqtSignal, pyqtSlot, QRunnable, QObject

from train_model import model_fit, model_predict, model_predict_proba, get_confusion_matrix, get_roc_curve, \
    get_precision_recall_curve, get_average_precision_score, get_learning_curve, get_validation_curve


class WorkerSignals(QObject):
    run_config = pyqtSignal(dict)
    progress = pyqtSignal(int)


class TrainWorker(QRunnable):
    def __init__(self, dataset_type, X, y, X_train, y_train, X_test, y_test, model, param_name, param_range, scoring):
        super().__init__()
        self.signals = WorkerSignals()

        self.dataset_type = dataset_type
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.param_name = param_name
        self.param_range = param_range
        self.scoring = scoring

    @pyqtSlot()
    def run(self):

        self.model = model_fit(self.model, self.X_train, self.y_train)
        predictions = model_predict(self.model, self.X_test)

        train_sizes, train_mean, val_mean = get_learning_curve(self.model, self.X, self.y)

        vc_train_mean, vc_test_mean = get_validation_curve(self.model, self.X, self.y, self.param_name,
                                                           self.param_range, self.scoring)

        if self.dataset_type == "Regression":

            self.signals.run_config.emit({
                "actual": self.y_test,
                "predictions": predictions,
                "lc_train_sizes": train_sizes,
                "lc_train_mean": train_mean,
                "lc_val_mean": val_mean,
                "param_range": self.param_range,
                "param_name": self.param_name,
                "scoring": self.scoring,
                "vc_train_mean": vc_train_mean,
                "vc_test_mean": vc_test_mean,
                "name": f"{self.model}"
            })

        elif self.dataset_type == "Classification":

            predictions_score = model_predict_proba(self.model, self.X_test)

            cm = get_confusion_matrix(self.y_test, predictions)

            fpr, tpr, roc_auc = get_roc_curve(self.y_test, predictions_score)

            precision, recall = get_precision_recall_curve(self.y_test, predictions_score)
            ap = get_average_precision_score(self.y_test, predictions_score)

            self.signals.run_config.emit({
                "cm": cm,
                "classes": [0, 1],
                "fpr": fpr,
                "tpr": tpr,
                "roc_auc": roc_auc,
                "lc_train_sizes": train_sizes,
                "lc_train_mean": train_mean,
                "lc_val_mean": val_mean,
                "y_true": self.y_test,
                "precision": precision,
                "recall": recall,
                "ap": ap,
                "param_range": self.param_range,
                "param_name": self.param_name,
                "scoring": self.scoring,
                "vc_train_mean": vc_train_mean,
                "vc_test_mean": vc_test_mean,
                "name": f"{self.model}"
            })

        self.signals.progress.emit(100)

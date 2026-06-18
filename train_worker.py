from PyQt6.QtCore import pyqtSignal, pyqtSlot, QRunnable, QObject

from train_model import model_fit, model_predict, model_predict_proba, get_confusion_matrix, get_roc_curve


class WorkerSignals(QObject):
    run_config = pyqtSignal(dict)
    progress = pyqtSignal(int)


class TrainWorker(QRunnable):
    def __init__(self, dataset_type, X_train, y_train, X_test, y_test, model):
        super().__init__()
        self.signals = WorkerSignals()

        self.dataset_type = dataset_type
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = model

    @pyqtSlot()
    def run(self):

        self.model = model_fit(self.model, self.X_train, self.y_train)
        predictions = model_predict(self.model, self.X_test)

        if self.dataset_type == "Regression":

            self.signals.run_config.emit({
                "actual": self.y_test,
                "predictions": predictions,
                "name": f"{self.model}"
            })

        elif self.dataset_type == "Classification":

            predictions_score = model_predict_proba(self.model, self.X_test)

            cm = get_confusion_matrix(self.y_test, predictions)

            fpr, tpr, roc_auc = get_roc_curve(self.y_test, predictions_score)

            self.signals.run_config.emit({
                "cm": cm,
                "classes": [0, 1],
                "fpr": fpr,
                "tpr": tpr,
                "roc_auc": roc_auc,
                "name": f"{self.model}"
            })

        self.signals.progress.emit(100)

from PyQt6.QtCore import pyqtSignal, pyqtSlot, QRunnable, QObject
import numpy as np

from train_model import model_fit, model_predict


class WorkerSignals(QObject):
    run_config = pyqtSignal(dict)


class TrainWorker(QRunnable):
    def __init__(self, X_train, y_train, X_test, y_test, model):
        super().__init__()
        self.signals = WorkerSignals()

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = model

    @pyqtSlot()
    def run(self):
        print("I am running in a thread")

        model = model_fit(self.model, self.X_train, self.y_train)
        predictions = model_predict(model, self.X_test)

        print(self.y_test.squeeze())
        print(predictions.squeeze())

        self.signals.run_config.emit({
            "x": np.arange(self.y_test.shape[0]),
            "y": self.y_test.squeeze(),
            "predictions": predictions.squeeze(),
            "title": "Predicted Plot",
            "name": "something"
        })

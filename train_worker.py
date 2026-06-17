from PyQt6.QtCore import pyqtSignal, pyqtSlot, QRunnable, QObject
import numpy as np


class WorkerSignals(QObject):
    run_config = pyqtSignal(dict)


class TrainWorker(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        print("I am running in a thread")

        self.signals.run_config.emit({
            "x": np.random.randint(1, 10, 5),
            "y": np.random.randint(10, 50, 5),
            "title": "New Run",
            "name": "something"
        })

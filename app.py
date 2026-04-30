import sys

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QComboBox, QLabel, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from train_model import *


# ----------------------------
# TRAIN THREAD
# ----------------------------
class TrainWorker(QThread):
    finished = pyqtSignal(object, object, object, object, object)
    progress = pyqtSignal(int)

    def __init__(self, dataset, lr, loss_name, optimizer_name, X_train, X_test, y_train, y_test):
        super().__init__()
        self.dataset = dataset
        self.lr = lr
        self.loss_name = loss_name
        self.optimizer_name = optimizer_name
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def run(self):
        model = make_model(self.dataset)

        loss_fn = get_loss_func(self.loss_name)
        opt = get_optimizer(self.optimizer_name, self.lr, model)

        if self.dataset == "Regression":
            epoch_count, train_loss, test_loss = model_train(model, loss_fn, opt, self.X_train, self.X_test,
                                                             self.y_train, self.y_test, self.progress.emit)

            self.finished.emit(epoch_count, train_loss, test_loss, -1, -1)

        elif self.dataset == "Classification":
            (epoch_count, train_loss, test_loss, train_accuracy,
             test_accuracy) = model_train(model, loss_fn, opt, self.X_train, self.X_test, self.y_train,
                                          self.y_test, self.progress.emit)

            self.finished.emit(epoch_count, train_loss, test_loss, train_accuracy, test_accuracy)


# ----------------------------
# MAIN WINDOW
# ----------------------------
class PlotWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ML Experiment Tracker")

        self.history = []
        self.current_run = None

        main_layout = QHBoxLayout()

        controls = QVBoxLayout()

        self.dataset = QComboBox()
        self.dataset.addItems(["Regression", "Classification"])
        controls.addWidget(QLabel("Dataset"))
        controls.addWidget(self.dataset)
        self.dataset.currentIndexChanged.connect(self.on_dataset_change)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.load_dataset()

        # ----------------------------
        # LEARNING RATE
        # ----------------------------
        self.lr_slider = QSlider(Qt.Orientation.Horizontal)
        self.lr_slider.setRange(0, 3)

        controls.addWidget(QLabel("Learning Rate"))
        controls.addWidget(self.lr_slider)

        # ----------------------------
        # LOSS FUNCTION
        # ----------------------------
        self.loss_box = QComboBox()
        self.loss_box.addItems(["Mean Squared Error", "Mean Absolute Error", "Huber Loss"])

        controls.addWidget(QLabel("Loss Function"))
        controls.addWidget(self.loss_box)

        # ----------------------------
        # OPTIMIZER
        # ----------------------------
        self.opt_box = QComboBox()
        self.opt_box.addItems(["Adam", "SGD", "RMSprop"])

        controls.addWidget(QLabel("Optimizer"))
        controls.addWidget(self.opt_box)

        # ----------------------------
        # PROGRESS BAR
        # ----------------------------
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)

        controls.addWidget(QLabel("Progress"))
        controls.addWidget(self.progress)

        # ----------------------------
        # COMPARE UI
        # ----------------------------
        self.compare_box = QComboBox()
        self.compare_box.addItem("Select run")

        self.compare_box.currentIndexChanged.connect(self.compare_runs)

        controls.addWidget(QLabel("Compare Runs"))
        controls.addWidget(self.compare_box)

        # ----------------------------
        # PLOTS
        # ----------------------------
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)

        self.fig2 = Figure()
        self.canvas2 = FigureCanvas(self.fig2)

        main_layout.addLayout(controls, 1)

        right = QVBoxLayout()
        right.addWidget(self.canvas)
        right.addWidget(self.canvas2)

        main_layout.addLayout(right, 3)

        self.setLayout(main_layout)

        # AUTO-TRAIN TRIGGERS
        self.lr_slider.valueChanged.connect(self.run_training)
        self.loss_box.currentIndexChanged.connect(self.run_training)
        self.opt_box.currentIndexChanged.connect(self.run_training)

    def load_dataset(self):
        X, y = get_data(self.dataset.currentText())

        X_train, X_test, y_train, y_test = make_train_test(X, y)
        X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def reset_ui(self):
        self.lr_slider.blockSignals(True)
        self.loss_box.blockSignals(True)
        self.opt_box.blockSignals(True)

        self.loss_box.clear()

        if self.dataset.currentText() == "Regression":
            self.loss_box.addItems(["Mean Squared Error", "Mean Absolute Error", "Huber Loss"])

        elif self.dataset.currentText() == "Classification":
            self.loss_box.addItems(["Binary Cross Entropy"])

        self.lr_slider.setValue(0)
        self.loss_box.setCurrentIndex(0)
        self.opt_box.setCurrentIndex(0)
        self.progress.setValue(0)

        self.lr_slider.blockSignals(False)
        self.loss_box.blockSignals(False)
        self.opt_box.blockSignals(False)

        self.compare_box.blockSignals(True)
        self.compare_box.clear()
        self.compare_box.addItem("Select run")
        self.compare_box.blockSignals(False)

        self.fig.clear()
        self.canvas.draw()

        self.fig2.clear()
        self.canvas2.draw()

    def on_dataset_change(self):
        self.history.clear()
        self.current_run = None

        self.setEnabled(False)

        self.load_dataset()
        self.reset_ui()

        self.setEnabled(True)

    # ----------------------------
    # VALUES
    # ----------------------------
    def lr(self):
        return [0.1, 0.01, 0.001, 0.0001][self.lr_slider.value()]

    def loss(self):
        return self.loss_box.currentText()

    def opt(self):
        return self.opt_box.currentText()

    # ----------------------------
    # UI LOCK
    # ----------------------------
    def set_ui(self, state):
        self.lr_slider.setEnabled(state)
        self.loss_box.setEnabled(state)
        self.opt_box.setEnabled(state)

    # ----------------------------
    # TRAIN
    # ----------------------------
    def run_training(self):
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()

        self.set_ui(False)
        self.progress.setValue(0)

        self.worker = TrainWorker(self.dataset.currentText(), self.lr(), self.loss(), self.opt(), self.X_train,
                                  self.X_test, self.y_train, self.y_test)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.save_run)
        self.worker.start()

    # ----------------------------
    # SAVE RUN
    # ----------------------------
    def save_run(self, epochs, train_loss, test_loss):
        new_run = {
            "epochs": epochs,
            "train": train_loss,
            "test": test_loss,
            "lr": self.lr(),
            "loss": self.loss(),
            "opt": self.opt(),
            "name": f"{self.loss()} {self.opt()} lr={self.lr()}"
        }

        # ----------------------------
        # MOVE OLD CURRENT INTO HISTORY
        # ----------------------------
        if self.current_run is not None:
            self.history.append(self.current_run)

            if len(self.history) > 3:
                self.history.pop(0)

        # ----------------------------
        # SET NEW CURRENT RUN
        # ----------------------------
        self.current_run = new_run

        self.update_dropdown()
        self.plot_current(new_run)

        self.set_ui(True)

    # ----------------------------
    # DROPDOWN
    # ----------------------------
    def update_dropdown(self):
        self.compare_box.clear()
        self.compare_box.addItem("Select run")

        for i, r in enumerate(self.history):
            self.compare_box.addItem(f"Run {i + 1}: {r['name']}")

    # ----------------------------
    # MAIN PLOT
    # ----------------------------
    def plot_current(self, run):
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        ax.plot(run["epochs"], run["train"], label="Train Loss")
        ax.plot(run["epochs"], run["test"], label="Test Loss")

        ax.set_title("Current Run")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")

        ax.legend()
        ax.grid()

        self.canvas.draw()

    # ----------------------------
    # COMPARE
    # ----------------------------
    def compare_runs(self):
        idx = self.compare_box.currentIndex() - 1
        if idx < 0 or idx >= len(self.history):
            return

        selected = self.history[idx]
        current = self.current_run

        self.fig2.clear()
        ax = self.fig2.add_subplot(111)

        ax.plot(current["epochs"], current["train"], label="Current Train Loss")
        ax.plot(current["epochs"], current["test"], label="Current Test Loss")

        ax.plot(selected["epochs"], selected["train"], "--", label="Selected Train Loss")
        ax.plot(selected["epochs"], selected["test"], "--", label="Selected Test Loss")

        ax.set_title("Comparison")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")

        ax.legend()
        ax.grid()

        self.canvas2.draw()


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlotWindow()
    window.showMaximized()
    sys.exit(app.exec())

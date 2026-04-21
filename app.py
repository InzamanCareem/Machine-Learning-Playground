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
    finished = pyqtSignal(object, object, object)
    progress = pyqtSignal(int)

    def __init__(self, lr, optimizer_name):
        super().__init__()
        self.lr = lr
        self.optimizer_name = optimizer_name

    def run(self):
        X, y = get_data()
        X_train, X_test, y_train, y_test = make_train_test(X, y)

        X_train, X_test, y_train, y_test = preprocess(
            X_train, X_test, y_train, y_test
        )

        model = make_model()
        loss_fn = loss_func()
        opt = get_optimizer(self.optimizer_name, self.lr, model)

        epoch_count, train_loss, test_loss = model_train(model, loss_fn, opt, X_train, X_test, y_train, y_test,
                                                   self.progress.emit)

        self.finished.emit(epoch_count, train_loss, test_loss)


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

        # ----------------------------
        # LEARNING RATE
        # ----------------------------
        self.lr_slider = QSlider(Qt.Orientation.Horizontal)
        self.lr_slider.setRange(0, 3)

        controls.addWidget(QLabel("Learning Rate"))
        controls.addWidget(self.lr_slider)

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

        self.compare_btn = QProgressBar()  # (dummy placeholder removed button completely)
        self.compare_btn = None

        self.compare_btn_real = QProgressBar()

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
        self.opt_box.currentIndexChanged.connect(self.run_training)

    # ----------------------------
    # VALUES
    # ----------------------------
    def lr(self):
        return [0.1, 0.01, 0.001, 0.0001][self.lr_slider.value()]

    def opt(self):
        return self.opt_box.currentText()

    # ----------------------------
    # UI LOCK
    # ----------------------------
    def set_ui(self, state):
        self.lr_slider.setEnabled(state)
        self.opt_box.setEnabled(state)

    # ----------------------------
    # TRAIN
    # ----------------------------
    def run_training(self):
        self.set_ui(False)
        self.progress.setValue(0)

        self.worker = TrainWorker(self.lr(), self.opt())
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
            "opt": self.opt(),
            "name": f"{self.opt()} lr={self.lr()}"
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

        ax.plot(run["epochs"], run["train"], label="Train")
        ax.plot(run["epochs"], run["test"], label="Test")

        ax.set_title("Current Run")

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

        ax.plot(current["epochs"], current["train"], label="Current Train")
        ax.plot(current["epochs"], current["test"], label="Current Test")

        ax.plot(selected["epochs"], selected["train"], "--", label="Selected Train")
        ax.plot(selected["epochs"], selected["test"], "--", label="Selected Test")

        ax.set_title("Comparison")

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

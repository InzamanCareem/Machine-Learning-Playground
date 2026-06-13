import sys

import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QComboBox, QLabel, QProgressBar, QTabWidget
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from train_worker import TrainWorker


class PlotWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ML Experiment Tracker")

        self.history = []
        self.current_run = None

        main_layout = QHBoxLayout()

        config_layout = QVBoxLayout()

        controls_layout = QVBoxLayout()

        control_tabs = QTabWidget()

        dataset_tab = QWidget()
        dataset_tab_layout = QVBoxLayout()

        model_tab = QWidget()
        model_tab_layout = QVBoxLayout()

        progress_layout = QVBoxLayout()

        self.dataset = QComboBox()
        self.dataset.addItems(["Regression", "Classification"])
        dataset_tab_layout.addWidget(QLabel("Dataset"))
        dataset_tab_layout.addWidget(self.dataset)
        self.dataset.currentIndexChanged.connect(self.on_dataset_change)

        self.samples_slider = QSlider(Qt.Orientation.Horizontal)
        self.samples_slider.setRange(0, 4)
        dataset_tab_layout.addWidget(QLabel("Number of Samples"))
        dataset_tab_layout.addWidget(self.samples_slider)

        self.features_slider = QSlider(Qt.Orientation.Horizontal)
        self.features_slider.setRange(0, 3)
        dataset_tab_layout.addWidget(QLabel("Number of Features"))
        dataset_tab_layout.addWidget(self.features_slider)

        # ----------------------------
        # LEARNING RATE
        # ----------------------------
        self.lr_slider = QSlider(Qt.Orientation.Horizontal)
        self.lr_slider.setRange(0, 15)
        start = 1
        end = 4
        self.lr_values = np.logspace(-start, -end, num=(end - start) * 5 + 1)

        model_tab_layout.addWidget(QLabel("Learning Rate"))
        model_tab_layout.addWidget(self.lr_slider)

        # ----------------------------
        # LOSS FUNCTION
        # ----------------------------
        self.loss_box = QComboBox()
        self.loss_box.addItems(["Mean Squared Error", "Mean Absolute Error", "Huber Loss"])

        model_tab_layout.addWidget(QLabel("Loss Function"))
        model_tab_layout.addWidget(self.loss_box)

        # ----------------------------
        # OPTIMIZER
        # ----------------------------
        self.opt_box = QComboBox()
        self.opt_box.addItems(["Adam", "SGD", "RMSprop"])

        model_tab_layout.addWidget(QLabel("Optimizer"))
        model_tab_layout.addWidget(self.opt_box)

        # ----------------------------
        # PROGRESS BAR
        # ----------------------------
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)

        progress_layout.addWidget(QLabel("Progress"))
        progress_layout.addWidget(self.progress)

        # ----------------------------
        # COMPARE UI
        # ----------------------------
        self.compare_box = QComboBox()
        self.compare_box.addItem("Select run")

        self.compare_box.currentIndexChanged.connect(self.compare_runs)

        progress_layout.addWidget(QLabel("Compare Runs"))
        progress_layout.addWidget(self.compare_box)

        dataset_tab.setLayout(dataset_tab_layout)
        model_tab.setLayout(model_tab_layout)

        control_tabs.addTab(dataset_tab, "Dataset")
        control_tabs.addTab(model_tab, "Model")

        controls_layout.addWidget(control_tabs)

        config_layout.addLayout(controls_layout, 3)
        config_layout.addLayout(progress_layout, 1)

        main_layout.addLayout(config_layout, 1)

        # ----------------------------
        # PLOTS
        # ----------------------------
        canvas_layout = QVBoxLayout()

        plot_tabs = QTabWidget()

        loss_plot_tab = QWidget()
        loss_plot_tab_layout = QVBoxLayout()

        accuracy_plot_tab = QWidget()
        accuracy_plot_tab_layout = QVBoxLayout()

        validation_curve_tab = QWidget()
        validation_curve_tab_layout = QVBoxLayout()

        learning_curve_tab = QWidget()
        learning_curve_tab_layout = QVBoxLayout()

        self.loss_plot = Figure()
        self.canvas = FigureCanvas(self.loss_plot)

        self.loss_compare_plot = Figure()
        self.canvas2 = FigureCanvas(self.loss_compare_plot)

        loss_plot_tab_layout.addWidget(self.canvas)
        loss_plot_tab_layout.addWidget(self.canvas2)

        loss_plot_tab.setLayout(loss_plot_tab_layout)

        plot_tabs.addTab(loss_plot_tab, "Loss Plot")
        plot_tabs.addTab(accuracy_plot_tab, "Accuracy Plot")
        plot_tabs.addTab(learning_curve_tab, "Learning Curve")
        plot_tabs.addTab(validation_curve_tab, "Validation Curve")

        canvas_layout.addWidget(plot_tabs)

        main_layout.addLayout(canvas_layout, 3)

        self.setLayout(main_layout)

        # AUTO-TRAIN TRIGGERS
        # TODO: add training config to file and use that, do not train every time
        self.samples_slider.valueChanged.connect(self.run_training)
        self.features_slider.valueChanged.connect(self.run_training)
        self.lr_slider.valueChanged.connect(self.run_training)
        self.loss_box.currentIndexChanged.connect(self.run_training)
        self.opt_box.currentIndexChanged.connect(self.run_training)

    def reset_ui(self):
        self.samples_slider.blockSignals(True)
        self.features_slider.blockSignals(True)
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

        self.samples_slider.blockSignals(False)
        self.features_slider.blockSignals(False)
        self.lr_slider.blockSignals(False)
        self.loss_box.blockSignals(False)
        self.opt_box.blockSignals(False)

        self.compare_box.blockSignals(True)
        self.compare_box.clear()
        self.compare_box.addItem("Select run")
        self.compare_box.blockSignals(False)

        self.loss_plot.clear()
        self.canvas.draw()

        self.loss_compare_plot.clear()
        self.canvas2.draw()

    def on_dataset_change(self):
        self.history.clear()
        self.current_run = None

        self.setEnabled(False)

        self.reset_ui()

        self.setEnabled(True)

    # ----------------------------
    # VALUES
    # ----------------------------
    def samples(self):
        return [100, 500, 1000, 5000, 10000][self.samples_slider.value()]

    def features(self):
        return [2, 4, 8, 16][self.features_slider.value()]

    def lr(self):
        return self.lr_values[self.lr_slider.value()]

    def loss(self):
        return self.loss_box.currentText()

    def opt(self):
        return self.opt_box.currentText()

    # ----------------------------
    # UI LOCK
    # ----------------------------
    def set_ui(self, state):
        self.samples_slider.setEnabled(state)
        self.features_slider.setEnabled(state)
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

        self.worker = TrainWorker(self.dataset.currentText(), self.samples(), self.features(), self.lr(), self.loss(),
                                  self.opt())
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
            "samples": self.samples(),
            "features": self.features(),
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
        self.loss_plot.clear()
        ax = self.loss_plot.add_subplot(111)

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

        self.loss_compare_plot.clear()
        ax = self.loss_compare_plot.add_subplot(111)

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

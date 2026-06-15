from PyQt6.QtWidgets import QVBoxLayout, QWidget, QTabWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from utils import plot_curves


class PlotsPanel:
    def __init__(self):
        self.canvas_layout = QVBoxLayout()

        self.plot_tabs = QTabWidget()

        self.loss_plot_tab = QWidget()
        self.loss_plot_tab_layout = QVBoxLayout()

        self.accuracy_plot_tab = QWidget()
        self.accuracy_plot_tab_layout = QVBoxLayout()

        self.learning_curve_tab = QWidget()
        self.learning_curve_tab_layout = QVBoxLayout()

        self.validation_curve_tab = QWidget()
        self.validation_curve_tab_layout = QVBoxLayout()

        self.loss_plot = Figure()
        self.loss_plot_canvas = FigureCanvas(self.loss_plot)

        self.loss_compare_plot = Figure()
        self.loss_plot_compare_canvas = FigureCanvas(self.loss_compare_plot)

        self.accuracy_plot = Figure()
        self.accuracy_plot_canvas = FigureCanvas(self.accuracy_plot)

        self.accuracy_compare_plot = Figure()
        self.accuracy_plot_compare_canvas = FigureCanvas(self.accuracy_compare_plot)

        self.learning_curve = Figure()
        self.learning_curve_canvas = FigureCanvas(self.learning_curve)

        self.validation_curve = Figure()
        self.validation_curve_canvas = FigureCanvas(self.validation_curve)

        self.loss_plot_tab_layout.addWidget(self.loss_plot_canvas)
        self.loss_plot_tab_layout.addWidget(self.loss_plot_compare_canvas)
        self.loss_plot_tab.setLayout(self.loss_plot_tab_layout)

        self.accuracy_plot_tab_layout.addWidget(self.accuracy_plot_canvas)
        self.accuracy_plot_tab_layout.addWidget(self.accuracy_plot_compare_canvas)
        self.accuracy_plot_tab.setLayout(self.accuracy_plot_tab_layout)

        self.learning_curve_tab_layout.addWidget(self.learning_curve_canvas)
        self.learning_curve_tab.setLayout(self.learning_curve_tab_layout)

        self.validation_curve_tab_layout.addWidget(self.validation_curve_canvas)
        self.validation_curve_tab.setLayout(self.validation_curve_tab_layout)

        self.plot_tabs.addTab(self.loss_plot_tab, "Loss Plot")
        self.plot_tabs.addTab(self.accuracy_plot_tab, "Accuracy Plot")
        self.plot_tabs.addTab(self.learning_curve_tab, "Learning Curve")
        self.plot_tabs.addTab(self.validation_curve_tab, "Validation Curve")

        self.canvas_layout.addWidget(self.plot_tabs)

    def plot_loss_curve(self, run):
        self.loss_plot.clear()
        ax = self.loss_plot.add_subplot(111)

        plot_curves(ax, run["epochs"], [(run["train_loss"], "Train Loss", "-"), (run["test_loss"], "Test Loss", "-")],
                    title="Current Run", x_label="Epochs", y_label="Loss")

        self.loss_plot_canvas.draw()

    def compare_loss_curves(self):
        idx = self.compare_box.currentIndex() - 1
        if idx < 0 or idx >= len(self.history):
            return

        selected = self.history[idx]
        current = self.current_run

        self.loss_compare_plot.clear()
        ax = self.loss_compare_plot.add_subplot(111)

        plot_curves(ax, current["epochs"],
                    [(current["train_loss"], "Current Train Loss", "-"),
                     (current["test_loss"], "Current Test Loss", "-"),
                     (selected["train_loss"], "Selected Train Loss", "--"),
                     (selected["test_loss"], "Selected Test Loss", "--")],
                    title="Comparison", x_label="Epochs", y_label="Loss")

        self.loss_plot_compare_canvas.draw()

    def plot_accuracy_curve(self, run):
        self.accuracy_plot.clear()
        ax = self.accuracy_plot.add_subplot(111)

        plot_curves(ax, run["epochs"],
                    [(run["train_accuracy"], "Train Accuracy", "-"), (run["test_accuracy"], "Test Accuracy", "-")],
                    title="Current Run", x_label="Epochs", y_label="Accuracy")

        self.accuracy_plot_canvas.draw()

    def compare_accuracy_curves(self):
        idx = self.compare_box.currentIndex() - 1
        if idx < 0 or idx >= len(self.history):
            return

        selected = self.history[idx]
        current = self.current_run

        self.accuracy_compare_plot.clear()
        ax = self.accuracy_compare_plot.add_subplot(111)

        plot_curves(ax, current["epochs"],
                    [(current["train_accuracy"], "Current Train Accuracy", "-"),
                     (current["test_accuracy"], "Current Test Accuracy", "-"),
                     (selected["train_accuracy"], "Selected Train Accuracy", "--"),
                     (selected["test_accuracy"], "Selected Test Accuracy", "--")],
                    title="Comparison", x_label="Epochs", y_label="Accuracy")

        self.accuracy_plot_compare_canvas.draw()

    def plot_learning_curve(self, run):
        self.learning_curve.clear()
        ax = self.learning_curve.add_subplot(111)

        plot_curves(ax, run["lc_train_sizes"],
                    [(run["lc_train_mean"], "Training", "-"),
                     (run["lc_val_mean"], "Validation", "-")],
                    title="Learning Curve", x_label="Training dataset size", y_label="Scoring")

        self.learning_curve_canvas.draw()

    def plot_validation_curve(self, run):
        self.validation_curve.clear()
        ax = self.validation_curve.add_subplot(111)

        # TODO: add marker
        plot_curves(ax, run["param_range"],
                    [(run["vc_train_mean"], "Training score", "-"),
                     (run["vc_test_mean"], "Cross-validation score", "-")],
                    title="Validation Curve", x_label="Hyperparameter", y_label="Scoring")

        self.validation_curve_canvas.draw()

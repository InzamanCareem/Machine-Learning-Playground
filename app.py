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
from utils import plot_curves


class PlotWindow(QWidget):
    def _load_linear_regression_parameters(self):
        pass

    def _load_decision_tree_regression_parameters(self):
        self.max_depth_slider.setRange(0, 3)
        self.max_depth_values = [3, 5, 10, 20]

        self.model_tab_parameter_layout.addWidget(QLabel("Max Depth"))
        self.model_tab_parameter_layout.addWidget(self.max_depth_slider)

        self.min_samples_split_slider.setRange(0, 2)
        self.min_samples_split_values = [2, 5, 10]

        self.model_tab_parameter_layout.addWidget(QLabel("Min Samples Split"))
        self.model_tab_parameter_layout.addWidget(self.min_samples_split_slider)

        self.min_samples_leaf_slider.setRange(0, 2)
        self.min_samples_leaf_values = [1, 2, 4]

        self.model_tab_parameter_layout.addWidget(QLabel("Min Samples Leaf"))
        self.model_tab_parameter_layout.addWidget(self.min_samples_leaf_slider)

    def _load_custom_neural_network_regressor_parameters(self):
        self.lr_slider.setRange(0, 15)
        start = 1
        end = 4
        self.lr_values = np.logspace(-start, -end, num=(end - start) * 5 + 1)

        self.model_tab_parameter_layout.addWidget(QLabel("Learning Rate"))
        self.model_tab_parameter_layout.addWidget(self.lr_slider)

        self.loss_box.addItems(["Mean Squared Error", "Mean Absolute Error", "Huber Loss"])

        self.model_tab_parameter_layout.addWidget(QLabel("Loss Function"))
        self.model_tab_parameter_layout.addWidget(self.loss_box)

        self.opt_box.addItems(["Adam", "SGD", "RMSprop"])

        self.model_tab_parameter_layout.addWidget(QLabel("Optimizer"))
        self.model_tab_parameter_layout.addWidget(self.opt_box)

    def clear_layout(self):
        while self.model_tab_parameter_layout.count():
            item = self.model_tab_parameter_layout.takeAt(0)

            widget = item.widget()
            if widget:
                widget.deleteLater()

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
        model_tab_model_layout = QVBoxLayout()
        self.model_tab_parameter_layout = QVBoxLayout()

        progress_layout = QVBoxLayout()

        self.dataset = QComboBox()
        self.dataset.addItems(["Regression", "Classification"])
        self.dataset.currentIndexChanged.connect(self.on_dataset_change)
        config_layout.addWidget(QLabel("Dataset"))
        config_layout.addWidget(self.dataset)

        self.samples_slider = QSlider(Qt.Orientation.Horizontal)
        self.samples_slider.setRange(0, 4)
        dataset_tab_layout.addWidget(QLabel("Number of Samples"))
        dataset_tab_layout.addWidget(self.samples_slider)

        self.features_slider = QSlider(Qt.Orientation.Horizontal)
        self.features_slider.setRange(0, 3)
        dataset_tab_layout.addWidget(QLabel("Number of Features"))
        dataset_tab_layout.addWidget(self.features_slider)

        # ----------------------------
        # Model Selection
        # ----------------------------
        self.model = QComboBox()
        self.model.addItems(["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor", "SVR",
                             "KNeighborsRegressor", "Custom Neural Network"])
        self.model.currentIndexChanged.connect(self.on_model_change)
        model_tab_model_layout.addWidget(QLabel("Model"))
        model_tab_model_layout.addWidget(self.model)

        # ----------------------------
        # HYPERPARAMETERS
        # ----------------------------

        self.max_depth_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_samples_split_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_samples_leaf_slider = QSlider(Qt.Orientation.Horizontal)

        self.lr_slider = QSlider(Qt.Orientation.Horizontal)
        self.loss_box = QComboBox()
        self.opt_box = QComboBox()

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

        progress_layout.addWidget(QLabel("Compare Runs"))
        progress_layout.addWidget(self.compare_box)

        dataset_tab.setLayout(dataset_tab_layout)
        model_tab_layout.addLayout(model_tab_model_layout)
        model_tab_layout.addStretch(0)
        model_tab_layout.addLayout(self.model_tab_parameter_layout, 1)

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

        learning_curve_tab = QWidget()
        learning_curve_tab_layout = QVBoxLayout()

        validation_curve_tab = QWidget()
        validation_curve_tab_layout = QVBoxLayout()

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

        loss_plot_tab_layout.addWidget(self.loss_plot_canvas)
        loss_plot_tab_layout.addWidget(self.loss_plot_compare_canvas)
        loss_plot_tab.setLayout(loss_plot_tab_layout)

        accuracy_plot_tab_layout.addWidget(self.accuracy_plot_canvas)
        accuracy_plot_tab_layout.addWidget(self.accuracy_plot_compare_canvas)
        accuracy_plot_tab.setLayout(accuracy_plot_tab_layout)

        learning_curve_tab_layout.addWidget(self.learning_curve_canvas)
        learning_curve_tab.setLayout(learning_curve_tab_layout)

        validation_curve_tab_layout.addWidget(self.validation_curve_canvas)
        validation_curve_tab.setLayout(validation_curve_tab_layout)

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

        self.max_depth_slider.valueChanged.connect(self.run_training)
        self.min_samples_split_slider.valueChanged.connect(self.run_training)
        self.min_samples_leaf_slider.valueChanged.connect(self.run_training)

        self.lr_slider.valueChanged.connect(self.run_training)
        self.loss_box.currentIndexChanged.connect(self.run_training)
        self.opt_box.currentIndexChanged.connect(self.run_training)

    def reset_ui(self):
        self.model.blockSignals(True)
        self.samples_slider.blockSignals(True)
        self.features_slider.blockSignals(True)

        self.model.clear()

        if self.dataset.currentText() == "Regression":
            self.model.addItems(["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor", "SVR",
                                 "KNeighborsRegressor", "Custom Neural Network"])
        elif self.dataset.currentText() == "Classification":
            self.model.addItems(["LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier", "SVC",
                                 "KNeighborsClassifier", "Custom Neural Network"])

        self.model.setCurrentIndex(0)
        self.progress.setValue(0)
        self.model.blockSignals(False)
        self.samples_slider.blockSignals(False)
        self.features_slider.blockSignals(False)

        self.compare_box.blockSignals(True)
        self.compare_box.clear()
        self.compare_box.addItem("Select run")
        self.compare_box.blockSignals(False)

        self.loss_plot.clear()
        self.loss_plot_canvas.draw()

        self.loss_compare_plot.clear()
        self.loss_plot_compare_canvas.draw()

        self.accuracy_plot.clear()
        self.accuracy_plot_canvas.draw()

        self.accuracy_compare_plot.clear()
        self.accuracy_plot_compare_canvas.draw()

        self.learning_curve.clear()
        self.learning_curve_canvas.draw()

        self.validation_curve.clear()
        self.validation_curve_canvas.draw()

    # TODO: on_dataset_type_change()
    def on_dataset_change(self):
        self.history.clear()
        self.current_run = None

        self.setEnabled(False)

        self.reset_ui()

        self.setEnabled(True)

    # TODO: on_dataset_change()

    def on_model_change(self):
        self.setEnabled(False)

        self.max_depth_slider.blockSignals(True)
        self.min_samples_split_slider.blockSignals(True)
        self.min_samples_leaf_slider.blockSignals(True)

        self.lr_slider.blockSignals(True)
        self.loss_box.blockSignals(True)
        self.opt_box.blockSignals(True)

        self.clear_layout()
        if self.model.currentText() == "LinearRegression":
            self._load_linear_regression_parameters()
        elif self.model.currentText() == "DecisionTreeRegressor":
            self._load_decision_tree_regression_parameters()
        elif self.model.currentText() == "Custom Neural Network Regressor":
            self._load_custom_neural_network_regressor_parameters()

        self.max_depth_slider.setValue(0)
        self.min_samples_split_slider.setValue(0)
        self.min_samples_leaf_slider.setValue(0)
        self.lr_slider.setValue(0)
        self.loss_box.setCurrentIndex(0)
        self.opt_box.setCurrentIndex(0)

        self.progress.setValue(0)

        self.max_depth_slider.blockSignals(False)
        self.min_samples_split_slider.blockSignals(False)
        self.min_samples_leaf_slider.blockSignals(False)

        self.lr_slider.blockSignals(False)
        self.loss_box.blockSignals(False)
        self.opt_box.blockSignals(False)

        self.setEnabled(True)

    # ----------------------------
    # VALUES
    # ----------------------------
    def samples(self):
        return [100, 500, 1000, 5000, 10000][self.samples_slider.value()]

    def features(self):
        return [2, 4, 8, 16][self.features_slider.value()]

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

        if self.model.currentText() == "Custom Neural Network":
            self.worker = TrainWorker.from_dl_model(self.dataset.currentText(), self.samples(), self.features(),
                                                    self.model.currentText(), self.lr_values[self.lr_slider.value()],
                                                    self.loss_box.currentText(), self.opt_box.currentText())
        else:
            self.worker = TrainWorker.from_ml_model(self.dataset.currentText(), self.samples(), self.features(),
                                                    self.model.currentText())

        self.worker.progress.connect(self.progress.setValue)
        self.worker.run_config.connect(self.save_run)
        self.worker.start()

    # ----------------------------
    # SAVE RUN
    # ----------------------------
    def save_run(self, run_config):
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
        self.current_run = run_config

        self.update_dropdown()

        if run_config["model_type"] == "dlr":
            self.compare_box.currentIndexChanged.connect(self.compare_loss_curves)
            self.plot_loss_curve(run_config)
        elif run_config["model_type"] == "dlc":
            self.compare_box.currentIndexChanged.connect(self.compare_loss_curves)
            self.compare_box.currentIndexChanged.connect(self.compare_accuracy_curves)
            self.plot_loss_curve(run_config)
            self.plot_accuracy_curve(run_config)
        else:
            self.plot_learning_curve(run_config)
            self.plot_validation_curve(run_config)

        self.set_ui(True)

    # ----------------------------
    # DROPDOWN
    # ----------------------------
    def update_dropdown(self):
        self.compare_box.clear()
        self.compare_box.addItem("Select run")

        for i, r in enumerate(self.history):
            self.compare_box.addItem(f"Run {i + 1}: {r['name']}")

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


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlotWindow()
    window.showMaximized()
    sys.exit(app.exec())

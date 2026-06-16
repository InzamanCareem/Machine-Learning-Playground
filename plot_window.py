from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QTabWidget, QLabel

from run_manager import RunManager
from dataset_controls import DatasetControls
from model_controls import ModelControls
from progress_panel import ProgressPanel
from plots_panel import PlotsPanel


class PlotWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ML Experiment Tracker")

        main_layout = QHBoxLayout()

        config_layout = QVBoxLayout()

        controls_layout = QVBoxLayout()

        control_tabs = QTabWidget()

        self.dc = DatasetControls()

        self.dc.dataset.currentIndexChanged.connect(self.on_dataset_change)
        config_layout.addWidget(QLabel("Dataset"))
        config_layout.addWidget(self.dc.dataset)

        self.mc = ModelControls()

        self.mc.model.currentIndexChanged.connect(self.on_model_change)

        control_tabs.addTab(self.dc.dataset_tab, "Dataset")
        control_tabs.addTab(self.mc.model_tab, "Model")

        controls_layout.addWidget(control_tabs)

        config_layout.addLayout(controls_layout, 3)

        self.prp = ProgressPanel()

        config_layout.addLayout(self.prp.progress_layout, 1)

        main_layout.addLayout(config_layout, 1)

        self.ptp = PlotsPanel()

        main_layout.addLayout(self.ptp.canvas_layout, 3)

        self.setLayout(main_layout)

        self.rm = RunManager(self.dc, self.mc, self.ptp)

        self.rm.run()

    def reset_ui(self):
        self.mc.model.blockSignals(True)
        self.dc.samples_slider.blockSignals(True)
        self.dc.features_slider.blockSignals(True)

        self.mc.model.clear()

        if self.dc.dataset.currentText() == "Regression":
            self.mc.model.addItems(["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor", "SVR",
                                    "KNeighborsRegressor", "Custom Neural Network"])
        elif self.dc.dataset.currentText() == "Classification":
            self.mc.model.addItems(["LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier", "SVC",
                                    "KNeighborsClassifier", "Custom Neural Network"])

        self.mc.model.setCurrentIndex(0)
        self.prp.progress.setValue(0)
        self.mc.model.blockSignals(False)
        self.dc.samples_slider.blockSignals(False)
        self.dc.features_slider.blockSignals(False)

        self.prp.compare_box.blockSignals(True)
        self.prp.compare_box.clear()
        self.prp.compare_box.addItem("Select run")
        self.prp.compare_box.blockSignals(False)

        self.ptp.loss_plot.clear()
        self.ptp.loss_plot_canvas.draw()

        self.ptp.loss_compare_plot.clear()
        self.ptp.loss_plot_compare_canvas.draw()

        self.ptp.accuracy_plot.clear()
        self.ptp.accuracy_plot_canvas.draw()

        self.ptp.accuracy_compare_plot.clear()
        self.ptp.accuracy_plot_compare_canvas.draw()

        self.ptp.learning_curve.clear()
        self.ptp.learning_curve_canvas.draw()

        self.ptp.validation_curve.clear()
        self.ptp.validation_curve_canvas.draw()

    def on_dataset_change(self):
        self.rm.history.clear()
        self.rm.current_run = None

        self.setEnabled(False)

        self.reset_ui()

        self.setEnabled(True)

    def on_model_change(self):
        self.setEnabled(False)

        self.mc.max_depth_slider.blockSignals(True)
        self.mc.min_samples_split_slider.blockSignals(True)
        self.mc.min_samples_leaf_slider.blockSignals(True)

        # self.mc.lr_slider.blockSignals(True)
        # self.mc.loss_box.blockSignals(True)
        # self.mc.opt_box.blockSignals(True)

        self.mc.clear_layout()
        if self.mc.model.currentText() == "LinearRegression":
            self.mc.load_linear_regression_parameters()
        elif self.mc.model.currentText() == "DecisionTreeRegressor":
            self.mc.load_decision_tree_regression_parameters()
        elif self.mc.model.currentText() == "Custom Neural Network Regressor":
            self.mc.load_custom_neural_network_regressor_parameters()

        self.mc.max_depth_slider.setValue(0)
        self.mc.min_samples_split_slider.setValue(0)
        self.mc.min_samples_leaf_slider.setValue(0)
        # self.mc.lr_slider.setValue(0)
        # self.mc.loss_box.setCurrentIndex(0)
        # self.mc.opt_box.setCurrentIndex(0)

        self.prp.progress.setValue(0)

        self.mc.max_depth_slider.blockSignals(False)
        self.mc.min_samples_split_slider.blockSignals(False)
        self.mc.min_samples_leaf_slider.blockSignals(False)

        # self.mc.lr_slider.blockSignals(False)
        # self.mc.loss_box.blockSignals(False)
        # self.mc.opt_box.blockSignals(False)

        self.setEnabled(True)

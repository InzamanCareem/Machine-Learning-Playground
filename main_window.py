from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QTabWidget

from run_manager import RunManager
from dataset_type_controls import DatasetTypeControls
from dataset_controls import DatasetControls
from model_controls import ModelControls
from plots_panel import PlotsPanel
from progress_panel import ProgressPanel


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ML Experiment Tracker")

        main_layout = QHBoxLayout()

        control_panel_layout = QVBoxLayout()
        canvas_layout = QVBoxLayout()

        controls_layout = QVBoxLayout()
        control_tabs = QTabWidget()
        canvas_tabs = QTabWidget()

        progress_layout = QVBoxLayout()

        self.plots_panel = PlotsPanel(canvas_tabs)

        self.progress_panel = ProgressPanel(progress_layout, self.plots_panel)

        self.dataset_type_controls = DatasetTypeControls(control_panel_layout, self.progress_panel, self.reset_ui)

        self.run_manager = RunManager(self.dataset_type_controls, self.plots_panel, self.progress_panel)

        self.dataset_controls = DatasetControls(control_tabs, self.run_manager)
        self.model_controls = ModelControls(control_tabs, self.run_manager)

        self.progress_panel.set_layout()

        controls_layout.addWidget(control_tabs)
        canvas_layout.addWidget(canvas_tabs)

        control_panel_layout.addLayout(controls_layout, 3)
        control_panel_layout.addLayout(progress_layout, 1)

        main_layout.addLayout(control_panel_layout, 1)
        main_layout.addLayout(canvas_layout, 3)

        self.setLayout(main_layout)

    def reset_ui(self, dataset_type):
        self.dataset_controls.reset_values(dataset_type)
        self.model_controls.reset_values(dataset_type)
        self.progress_panel.reset_values()
        self.plots_panel.reset_values()

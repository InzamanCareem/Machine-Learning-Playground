from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSlider, QLabel, QWidget, QVBoxLayout, QComboBox

from train_worker import TrainWorker


class ModelControls:
    def __init__(self, control_tabs, threadpool, plot_panel, progress_panel):
        self.control_tabs = control_tabs
        self.threadpool = threadpool
        self.plot_panel = plot_panel
        self.progress_panel = progress_panel

        self.tab = QWidget()
        self.tab_layout = QVBoxLayout()

        self.model_tab_layout = QVBoxLayout()
        self.model_parameter_tab_layout = QVBoxLayout()

        self.model = QComboBox()
        self.model.addItems(["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "SVR",
                             "KNeighbors Regressor", "Custom Neural Network Regressor"])
        # self.model.currentIndexChanged.connect(self.on_model_change)

        self.depth_slider = QSlider(Qt.Orientation.Horizontal)
        self.depth_slider.setRange(0, 2)
        # self.depth_slider.valueChanged.connect(self.on_model_change)

        self._set_tab()

    def _set_tab(self):
        self.model_tab_layout.addWidget(QLabel("Model"))
        self.model_tab_layout.addWidget(self.model)

        self.model_parameter_tab_layout.addWidget(QLabel("Depth Slider"))
        self.model_parameter_tab_layout.addWidget(self.depth_slider)

        self.tab_layout.addLayout(self.model_tab_layout)
        self.tab_layout.addStretch(0)
        self.tab_layout.addLayout(self.model_parameter_tab_layout, 1)

        self.tab.setLayout(self.tab_layout)

        self.control_tabs.addTab(self.tab, "Model")

    def reset_values(self, dataset_type):
        self.model.clear()
        if dataset_type == "Regression":
            self.model.addItems(["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "SVR",
                             "KNeighbors Regressor", "Custom Neural Network Regressor"])
        elif dataset_type == "Classification":
            self.model.addItems((["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier", "SVC",
                                 "KNeighbors Classifier", "Custom Neural Network Classifier"]))

        self.model.setCurrentIndex(0)

    def on_model_change(self):
        print("This is the model")

        train_worker = TrainWorker()

        train_worker.signals.run_config.connect(self.plot_panel.plot_curve)
        train_worker.signals.run_config.connect(self.progress_panel.save_run)

        self.threadpool.start(train_worker)

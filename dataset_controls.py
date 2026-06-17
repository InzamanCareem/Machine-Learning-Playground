from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSlider, QLabel, QWidget, QVBoxLayout, QComboBox
from train_worker import TrainWorker


class DatasetControls:
    def __init__(self, control_tabs, threadpool, plot_panel, progress_panel):
        self.control_tabs = control_tabs
        self.threadpool = threadpool
        self.plot_panel = plot_panel
        self.progress_panel = progress_panel

        self.tab = QWidget()
        self.tab_layout = QVBoxLayout()

        self.dataset_tab_layout = QVBoxLayout()
        self.dataset_parameter_tab_layout = QVBoxLayout()

        self.dataset = QComboBox()
        self.dataset.addItems(["Custom regression dataset", "California housing dataset"])

        self.samples_slider = QSlider(Qt.Orientation.Horizontal)
        self.samples_slider.setRange(0, 4)
        self.samples_slider.valueChanged.connect(self.on_dataset_change)

        self.features_slider = QSlider(Qt.Orientation.Horizontal)
        self.features_slider.setRange(0, 3)
        self.features_slider.valueChanged.connect(self.on_dataset_change)

        self._set_tab()

    def _set_tab(self):
        self.dataset_tab_layout.addWidget(QLabel("Dataset"))
        self.dataset_tab_layout.addWidget(self.dataset)

        self.dataset_parameter_tab_layout.addWidget(QLabel("Number of Samples"))
        self.dataset_parameter_tab_layout.addWidget(self.samples_slider)

        self.dataset_parameter_tab_layout.addWidget(QLabel("Number of Features"))
        self.dataset_parameter_tab_layout.addWidget(self.features_slider)

        self.tab_layout.addLayout(self.dataset_tab_layout)
        self.tab_layout.addStretch(0)
        self.tab_layout.addLayout(self.dataset_parameter_tab_layout, 1)

        self.tab.setLayout(self.tab_layout)

        self.control_tabs.addTab(self.tab, "Dataset")

    def reset_values(self, dataset_type):
        self.dataset.clear()
        if dataset_type == "Regression":
            self.dataset.addItems(["Custom regression dataset", "California housing dataset"])
        elif dataset_type == "Classification":
            self.dataset.addItems(["Custom classification dataset", "Diabetes dataset"])

        self.dataset.setCurrentIndex(0)
        self.samples_slider.setValue(0)
        self.features_slider.setValue(0)

    # def samples(self):
    #     return [100, 500, 1000][self.samples_slider.value()]

    def on_dataset_change(self):
        print("This is the dataset")

        train_worker = TrainWorker()

        train_worker.signals.run_config.connect(self.plot_panel.plot_curve)
        train_worker.signals.run_config.connect(self.progress_panel.save_run)

        self.threadpool.start(train_worker)

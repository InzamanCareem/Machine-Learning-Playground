from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSlider, QLabel, QWidget, QVBoxLayout, QComboBox


class DatasetControls:
    def __init__(self, control_tabs, run_manager):
        self.control_tabs = control_tabs
        self.run_manager = run_manager

        self.tab = QWidget()
        self.tab_layout = QVBoxLayout()

        self.dataset_type = "Regression"

        self.dataset_tab_layout = QVBoxLayout()
        self.dataset_parameter_tab_layout = QVBoxLayout()

        self.dataset = QComboBox()
        self.dataset.addItems(["Custom regression dataset", "California housing dataset"])

        self.samples_slider = QSlider(Qt.Orientation.Horizontal)
        self.samples_slider.setRange(0, 4)
        self.samples_slider.valueChanged.connect(self.on_dataset_change)
        self.samples_slider_label = QLabel(f"Samples: {self.samples()}")

        self.features_slider = QSlider(Qt.Orientation.Horizontal)
        self.features_slider.setRange(0, 3)
        self.features_slider.valueChanged.connect(self.on_dataset_change)
        self.features_slider_label = QLabel(f"Features: {self.features()}")

        self.noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_slider.setRange(0, 6)
        self.noise_slider.valueChanged.connect(self.on_dataset_change)
        self.noise_slider_label = QLabel(f"Noise: {self.noise()}")

        self._set_tab()

        self.run_manager.load_dataset(self.samples(), self.features(), self.noise())

    def _set_tab(self):
        self.dataset_tab_layout.addWidget(QLabel("Dataset"))
        self.dataset_tab_layout.addWidget(self.dataset)

        self.dataset_parameter_tab_layout.addWidget(QLabel("Number of Samples"))
        self.dataset_parameter_tab_layout.addWidget(self.samples_slider)
        self.dataset_parameter_tab_layout.addWidget(self.samples_slider_label)

        self.dataset_parameter_tab_layout.addWidget(QLabel("Number of Features"))
        self.dataset_parameter_tab_layout.addWidget(self.features_slider)
        self.dataset_parameter_tab_layout.addWidget(self.features_slider_label)

        self.dataset_parameter_tab_layout.addWidget(QLabel("Noise"))
        self.dataset_parameter_tab_layout.addWidget(self.noise_slider)
        self.dataset_parameter_tab_layout.addWidget(self.noise_slider_label)

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
        self.noise_slider.setValue(0)

        self.dataset_type = dataset_type

    def samples(self):
        return [100, 500, 1000, 5000, 10000][self.samples_slider.value()]

    def features(self):
        return [2, 4, 6, 8][self.features_slider.value()]

    def noise(self):
        return [0, 0.5, 1, 2, 4, 8, 10][self.noise_slider.value()]

    def on_dataset_change(self):
        self.samples_slider_label.setText(f"Samples: {self.samples()}")
        self.features_slider_label.setText(f"Features: {self.features()}")
        self.noise_slider_label.setText(f"Noise: {self.noise()}")

        self.run_manager.load_dataset(self.samples(), self.features(), self.noise())

        self.run_manager.start()

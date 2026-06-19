from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSlider, QLabel, QWidget, QVBoxLayout, QComboBox, QHBoxLayout


class DatasetControls:
    REGRESSION_DATASETS = ["Custom regression dataset"]
    CLASSIFICATION_DATASETS = ["Custom classification dataset"]

    def __init__(self, control_tabs, run_manager):
        self.control_tabs = control_tabs
        self.run_manager = run_manager

        self.tab = QWidget()
        self.tab_layout = QVBoxLayout()

        self.dataset_tab_layout = QVBoxLayout()
        self.dataset_parameter_tab_layout = QVBoxLayout()

        self.dataset = QComboBox()
        self.dataset.addItems(DatasetControls.REGRESSION_DATASETS)
        self.dataset.currentIndexChanged.connect(self.on_dataset_change)

        self.current_parameters = []

        self._set_tab()

        self.run_manager.load_dataset(self.samples(), self.features(), self.noise())

    def _set_tab(self):
        self.dataset_tab_layout.addWidget(QLabel("Dataset"))
        self.dataset_tab_layout.addWidget(self.dataset)

        self._add_custom_dataset_parameters()

        self.tab_layout.addLayout(self.dataset_tab_layout)
        self.tab_layout.addStretch(0)
        self.tab_layout.addLayout(self.dataset_parameter_tab_layout, 1)

        self.tab.setLayout(self.tab_layout)

        self.control_tabs.addTab(self.tab, "Dataset")

    def _add_widgets(self, parameters):
        for parameter in parameters:
            parameter["parameter_value_layout"].addWidget(parameter["parameter"], 5)
            parameter["parameter_value_layout"].addWidget(parameter["parameter_value_label"], 1)
            parameter["parameter_value_layout"].setContentsMargins(0, 10, 0, 0)

            parameter["parameter_layout"].addWidget(parameter["label"])
            parameter["parameter_layout"].addLayout(parameter["parameter_value_layout"])
            parameter["parameter_layout"].addStretch(0)
            parameter["parameter_layout"].setContentsMargins(10, 10, 0, 0)

            self.dataset_parameter_tab_layout.addLayout(parameter["parameter_layout"])

    def reset_values(self, dataset_type):
        self.dataset.clear()
        if dataset_type == "Regression":
            self.dataset.addItems(DatasetControls.REGRESSION_DATASETS)
        elif dataset_type == "Classification":
            self.dataset.addItems(DatasetControls.CLASSIFICATION_DATASETS)

        self.samples_slider.blockSignals(True)
        self.features_slider.blockSignals(True)
        self.noise_slider.blockSignals(True)

        self.dataset.setCurrentIndex(0)
        self.samples_slider.setValue(0)
        self.features_slider.setValue(0)
        self.noise_slider.setValue(0)

        self.samples_slider.blockSignals(False)
        self.features_slider.blockSignals(False)
        self.noise_slider.blockSignals(False)

        self.run_manager.load_dataset(self.samples(), self.features(), self.noise())

    def _add_custom_dataset_parameters(self):
        self.samples_layout = QVBoxLayout()
        self.samples_slider_layout = QHBoxLayout()
        self.samples_slider_label = QLabel("Number of Samples")
        self.samples_slider = QSlider(Qt.Orientation.Horizontal)
        self.samples_slider.setRange(0, 4)
        self.samples_slider.valueChanged.connect(self.on_dataset_change)
        self.samples_value_label = QLabel(f"{self.samples()}")

        self.features_layout = QVBoxLayout()
        self.features_slider_layout = QHBoxLayout()
        self.features_slider_label = QLabel("Number of Features")
        self.features_slider = QSlider(Qt.Orientation.Horizontal)
        self.features_slider.setRange(0, 3)
        self.features_slider.valueChanged.connect(self.on_dataset_change)
        self.features_value_label = QLabel(f"{self.features()}")

        self.noise_layout = QVBoxLayout()
        self.noise_slider_layout = QHBoxLayout()
        self.noise_slider_label = QLabel("Noise")
        self.noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_slider.setRange(0, 6)
        self.noise_slider.valueChanged.connect(self.on_dataset_change)
        self.noise_value_label = QLabel(f"{self.noise()}")

        self.current_parameters = [{"label": self.samples_slider_label,
                                    "parameter": self.samples_slider,
                                    "parameter_layout": self.samples_layout,
                                    "parameter_value_layout": self.samples_slider_layout,
                                    "parameter_value_label": self.samples_value_label,
                                    "parameter_value": self.samples
                                    },

                                   {"label": self.features_slider_label,
                                    "parameter": self.features_slider,
                                    "parameter_layout": self.features_layout,
                                    "parameter_value_layout": self.features_slider_layout,
                                    "parameter_value_label": self.features_value_label,
                                    "parameter_value": self.features
                                    },

                                   {"label": self.noise_slider_label,
                                    "parameter": self.noise_slider,
                                    "parameter_layout": self.noise_layout,
                                    "parameter_value_layout": self.noise_slider_layout,
                                    "parameter_value_label": self.noise_value_label,
                                    "parameter_value": self.noise
                                    }
                                   ]

        self._add_widgets(self.current_parameters)

    @staticmethod
    def samples_range():
        return [100, 500, 1000, 5000, 10000]

    def samples(self):
        return self.samples_range()[self.samples_slider.value()]

    @staticmethod
    def features_range():
        return [2, 4, 6, 8]

    def features(self):
        return self.features_range()[self.features_slider.value()]

    @staticmethod
    def noise_range():
        return [0, 0.5, 1, 2, 4, 8, 10]

    def noise(self):
        return self.noise_range()[self.noise_slider.value()]

    def _update_parameter_labels(self):
        for parameter in self.current_parameters:
            parameter["parameter_value_label"].setText(f"{parameter["parameter_value"]()}")

    def on_dataset_change(self):

        self._update_parameter_labels()

        self.run_manager.load_dataset(self.samples(), self.features(), self.noise())

        self.run_manager.start()

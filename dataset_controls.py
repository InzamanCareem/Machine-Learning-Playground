from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QLabel, QSlider, QVBoxLayout, QWidget


class DatasetControls:
    def __init__(self):
        self.dataset_tab = QWidget()
        self.dataset_tab_layout = QVBoxLayout()

        self.dataset = QComboBox()
        self.dataset.addItems(["Regression", "Classification"])
        # self.dataset.currentIndexChanged.connect(self.on_dataset_change)
        # config_layout.addWidget(QLabel("Dataset"))
        # config_layout.addWidget(self.dataset)

        self.samples_slider = QSlider(Qt.Orientation.Horizontal)
        self.samples_slider.setRange(0, 4)
        self.dataset_tab_layout.addWidget(QLabel("Number of Samples"))
        self.dataset_tab_layout.addWidget(self.samples_slider)

        self.features_slider = QSlider(Qt.Orientation.Horizontal)
        self.features_slider.setRange(0, 3)
        self.dataset_tab_layout.addWidget(QLabel("Number of Features"))
        self.dataset_tab_layout.addWidget(self.features_slider)

        self.dataset_tab.setLayout(self.dataset_tab_layout)

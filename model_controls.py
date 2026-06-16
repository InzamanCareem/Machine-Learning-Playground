import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QLabel, QSlider, QWidget, QVBoxLayout


class ModelControls:
    def __init__(self):
        self.model_tab = QWidget()
        self.model_tab_layout = QVBoxLayout()
        self.model_tab_model_layout = QVBoxLayout()
        self.model_tab_parameter_layout = QVBoxLayout()

        self.model = QComboBox()
        self.model.addItems(["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor", "SVR",
                             "KNeighborsRegressor", "Custom Neural Network"])

        # self.model.currentIndexChanged.connect(self.on_model_change)
        self.model_tab_model_layout.addWidget(QLabel("Model"))
        self.model_tab_model_layout.addWidget(self.model)

        self.model_tab_layout.addLayout(self.model_tab_model_layout)
        self.model_tab_layout.addStretch(0)
        self.model_tab_layout.addLayout(self.model_tab_parameter_layout, 1)

        self.model_tab.setLayout(self.model_tab_layout)

        # ----------------------------
        # HYPERPARAMETERS
        # ----------------------------

        self.max_depth_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_samples_split_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_samples_leaf_slider = QSlider(Qt.Orientation.Horizontal)

        self.lr_slider = QSlider(Qt.Orientation.Horizontal)
        self.loss_box = QComboBox()
        self.opt_box = QComboBox()

    def get_model(self):
        return self.model

    def get_controls(self):
        return [self.max_depth_slider, self.min_samples_split_slider, self.min_samples_leaf_slider]

    def load_linear_regression_parameters(self):
        pass

    def load_decision_tree_regression_parameters(self):
        self.max_depth_slider.setRange(0, 3)
        max_depth_values = [3, 5, 10, 20]

        self.model_tab_parameter_layout.addWidget(QLabel("Max Depth"))
        self.model_tab_parameter_layout.addWidget(self.max_depth_slider)

        self.min_samples_split_slider.setRange(0, 2)
        min_samples_split_values = [2, 5, 10]

        self.model_tab_parameter_layout.addWidget(QLabel("Min Samples Split"))
        self.model_tab_parameter_layout.addWidget(self.min_samples_split_slider)

        self.min_samples_leaf_slider.setRange(0, 2)
        min_samples_leaf_values = [1, 2, 4]

        self.model_tab_parameter_layout.addWidget(QLabel("Min Samples Leaf"))
        self.model_tab_parameter_layout.addWidget(self.min_samples_leaf_slider)

    def load_custom_neural_network_regressor_parameters(self):
        self.lr_slider.setRange(0, 15)
        start = 1
        end = 4
        lr_values = np.logspace(-start, -end, num=(end - start) * 5 + 1)

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

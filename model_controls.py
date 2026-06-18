import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSlider, QLabel, QWidget, QVBoxLayout, QComboBox


class ModelControls:
    def __init__(self, control_tabs, run_manager):
        self.control_tabs = control_tabs
        self.run_manager = run_manager

        self.tab = QWidget()
        self.tab_layout = QVBoxLayout()

        self.model_tab_layout = QVBoxLayout()
        self.model_parameter_tab_layout = QVBoxLayout()

        self.model = QComboBox()
        self.model.addItems(["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "SVR",
                             "KNeighbors Regressor", "Custom Neural Network Regressor"])
        self.model.currentIndexChanged.connect(self.on_model_change)

        self.current_parameters = []

        self._set_tab()

        self.run_manager.load_model(self.model.currentText(),
                                    **{parameter["alias"]: parameter["parameter_value"]() for parameter in
                                       self.current_parameters})

    def _set_tab(self):
        self.model_tab_layout.addWidget(QLabel("Model"))
        self.model_tab_layout.addWidget(self.model)

        self._add_linear_regression_parameters()

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

    def _clear_layout(self):
        while self.model_parameter_tab_layout.count():
            item = self.model_parameter_tab_layout.takeAt(0)

            widget = item.widget()
            if widget:
                widget.deleteLater()

    def _add_tabs(self, parameters):
        for parameter in parameters:
            self.model_parameter_tab_layout.addWidget(parameter["label"])
            self.model_parameter_tab_layout.addWidget(parameter["parameter"])
            self.model_parameter_tab_layout.addWidget(parameter["parameter_value_label"])

    def _add_linear_regression_parameters(self):
        self.tolerance_slider_label = QLabel(f"Tolerance")
        self.tolerance_slider = QSlider(Qt.Orientation.Horizontal)
        self.tolerance_slider.setRange(0, 9)
        self.tolerance_slider.valueChanged.connect(self.on_model_parameter_change)
        self.tolerance_slider_value_label = QLabel(f"Tolerance: {self.tolerance()}")

        self.current_parameters = [{"label": self.tolerance_slider_label, "parameter": self.tolerance_slider,
                                    "parameter_value_label": self.tolerance_slider_value_label,
                                    "parameter_value": self.tolerance, "alias": "tol"}]

        self._add_tabs(self.current_parameters)

    def _add_decision_tree_parameters(self):
        self.max_depth_slider_label = QLabel(f"Max Depth")
        self.max_depth_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_depth_slider.setRange(0, 3)
        self.max_depth_slider.valueChanged.connect(self.on_model_parameter_change)
        self.max_depth_slider_value_label = QLabel(f"Max Depth: {self.max_depth()}")

        self.min_samples_split_slider_label = QLabel(f"Min Samples Split")
        self.min_samples_split_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_samples_split_slider.setRange(0, 2)
        self.min_samples_split_slider.valueChanged.connect(self.on_model_parameter_change)
        self.min_samples_split_slider_value_label = QLabel(f"Min Samples Split: {self.min_samples_split()}")

        self.current_parameters = [{"label": self.max_depth_slider_label, "parameter": self.max_depth_slider,
                                    "parameter_value_label": self.max_depth_slider_value_label,
                                    "parameter_value": self.max_depth, "alias": "max_depth"},

                                   {"label": self.min_samples_split_slider_label,
                                    "parameter": self.min_samples_split_slider,
                                    "parameter_value_label": self.min_samples_split_slider_value_label,
                                    "parameter_value": self.min_samples_split, "alias": "min_samples_split"}
                                   ]

        self._add_tabs(self.current_parameters)

    def _add_logistic_regression_parameters(self):
        self.tolerance_slider_label = QLabel(f"Tolerance")
        self.tolerance_slider = QSlider(Qt.Orientation.Horizontal)
        self.tolerance_slider.setRange(0, 9)
        self.tolerance_slider.valueChanged.connect(self.on_model_parameter_change)
        self.tolerance_slider_value_label = QLabel(f"Tolerance: {self.tolerance()}")

        self.current_parameters = [{"label": self.tolerance_slider_label, "parameter": self.tolerance_slider,
                                    "parameter_value_label": self.tolerance_slider_value_label,
                                    "parameter_value": self.tolerance, "alias": "tol"}]

        self._add_tabs(self.current_parameters)

    def on_model_change(self):
        self._clear_layout()

        if self.model.currentText() == "Linear Regression":
            self._add_linear_regression_parameters()
        elif self.model.currentText() == "Decision Tree Regressor":
            self._add_decision_tree_parameters()
        elif self.model.currentText() == "Logistic Regression":
            self._add_logistic_regression_parameters()
        elif self.model.currentText() == "Decision Tree Classifier":
            self._add_decision_tree_parameters()

        self.run_manager.load_model(self.model.currentText(),
                                    **{parameter["alias"]: parameter["parameter_value"]() for parameter in
                                       self.current_parameters})

    def tolerance(self):
        return np.linspace(0.0001, 0.000001, 10)[self.tolerance_slider.value()]

    def max_depth(self):
        return [1, 5, 10, 20][self.max_depth_slider.value()]

    def min_samples_split(self):
        return [2, 5, 10][self.min_samples_split_slider.value()]

    def _update_parameter_labels(self):
        for parameter in self.current_parameters:
            parameter["parameter_value_label"].setText(f"{parameter["label"].text()}: {parameter["parameter_value"]()}")

    def on_model_parameter_change(self):
        self._update_parameter_labels()

        parameters = {parameter["alias"]: parameter["parameter_value"]() for parameter in self.current_parameters}

        print(self.model.currentText(), " on model controls")

        self.run_manager.load_model(self.model.currentText(), **parameters)

        self.run_manager.start()

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSlider, QLabel, QWidget, QVBoxLayout, QComboBox, QHBoxLayout


class ModelControls:
    REGRESSION_MODELS = ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor",
                         "Support Vector Regressor", "KNeighbors Regressor", "Custom Neural Network Regressor"]

    REGRESSION_SCORING = ["Select Scoring", "r2", "neg_mean_absolute_error", "neg_mean_squared_error",
                          "neg_root_mean_squared_error", "neg_mean_absolute_percentage_error"]

    CLASSIFICATION_MODELS = ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier",
                             "Support Vector Classifier", "KNeighbors Classifier", "Custom Neural Network Classifier"]

    CLASSIFICATION_SCORING = ["Select Scoring", "accuracy", "precision", "recall", "f1", "roc_auc", "neg_log_loss"]

    def __init__(self, control_tabs, run_manager):
        self.control_tabs = control_tabs
        self.run_manager = run_manager

        self.tab = QWidget()
        self.tab_layout = QVBoxLayout()

        self.model_tab_layout = QVBoxLayout()
        self.model_parameter_tab_layout = QVBoxLayout()

        self.model = QComboBox()
        self.model.addItems(ModelControls.REGRESSION_MODELS)
        self.model.currentIndexChanged.connect(self.on_model_change)

        self.model_parameter_selection_layout = QVBoxLayout()

        self.model_parameter_selection = QComboBox()
        self.model_parameter_selection.addItem("Select Parameter")
        self.model_parameter_selection.currentIndexChanged.connect(self.on_model_parameter_selection_change)

        self.model_scoring_selection = QComboBox()
        self.model_scoring_selection.addItems(ModelControls.REGRESSION_SCORING)
        self.model_scoring_selection.currentIndexChanged.connect(self.on_model_parameter_selection_change)

        self.current_parameters = []

        self._set_tab()

        self.run_manager.set_model_controls_ui(self._set_ui)

        self.run_manager.load_model(self.model.currentText(), **self._get_param_dict())

    def _set_tab(self):
        self.model_tab_layout.addWidget(QLabel("Model"))
        self.model_tab_layout.addWidget(self.model)

        self.model_parameter_selection_layout.addWidget(QLabel("Model Parameter Selection"))
        self.model_parameter_selection_layout.addWidget(self.model_parameter_selection)
        self.model_parameter_selection_layout.addWidget(self.model_scoring_selection)

        self._add_linear_logistic_regression_parameters()

        self.tab_layout.addLayout(self.model_tab_layout)
        self.tab_layout.addStretch(0)
        self.tab_layout.addLayout(self.model_parameter_tab_layout, 2)
        self.tab_layout.addLayout(self.model_parameter_selection_layout, 1)

        self.tab.setLayout(self.tab_layout)

        self.control_tabs.addTab(self.tab, "Model")

    def _set_ui(self, state):
        for parameter in self.current_parameters:
            parameter["parameter"].setEnabled(state)

    def reset_values(self, dataset_type):
        self.model.clear()

        self.model_parameter_selection.blockSignals(True)
        self.model_scoring_selection.blockSignals(True)

        self.model_scoring_selection.clear()

        if dataset_type == "Regression":
            self.model.addItems(ModelControls.REGRESSION_MODELS)
            self.model_scoring_selection.addItems(ModelControls.REGRESSION_SCORING)

        elif dataset_type == "Classification":
            self.model.addItems(ModelControls.CLASSIFICATION_MODELS)
            self.model_scoring_selection.addItems(ModelControls.CLASSIFICATION_SCORING)

        self.model.setCurrentIndex(0)
        self.model_parameter_selection.setCurrentIndex(0)
        self.model_scoring_selection.setCurrentIndex(0)

        self.model_parameter_selection.blockSignals(False)
        self.model_scoring_selection.blockSignals(False)

    def _clear_layout(self):
        while self.model_parameter_tab_layout.count():
            item = self.model_parameter_tab_layout.takeAt(0)

            stack = []

            if item.layout():
                stack.append(item.layout())

            if item.widget():
                item.widget().deleteLater()

            while stack:
                layout = stack.pop()

                while layout.count():
                    sub_item = layout.takeAt(0)

                    if sub_item.widget():
                        sub_item.widget().deleteLater()

                    if sub_item.layout():
                        stack.append(sub_item.layout())

                layout.deleteLater()

    def _add_widgets(self, parameters):
        for parameter in parameters:
            parameter["parameter_value_layout"].addWidget(parameter["parameter"], 5)

            parameter_value_label = parameter.get("parameter_value_label")
            if parameter_value_label is not None:
                parameter["parameter_value_layout"].addWidget(parameter_value_label, 1)

            parameter["parameter_layout"].addWidget(parameter["label"])
            parameter["parameter_layout"].addLayout(parameter["parameter_value_layout"])
            parameter["parameter_layout"].addStretch(0)

            self.model_parameter_tab_layout.addLayout(parameter["parameter_layout"])

    def _add_values(self, parameters):
        self.model_parameter_selection.blockSignals(True)
        self.model_parameter_selection.clear()
        self.model_parameter_selection.addItem("Select Parameter")

        for parameter in parameters:
            self.model_parameter_selection.addItem(parameter["alias"])

        self.model_parameter_selection.setCurrentIndex(0)
        self.model_parameter_selection.blockSignals(False)

    def _get_param_dict(self):
        return {parameter["alias"]: parameter["parameter_value"]() for parameter in self.current_parameters}

    def _add_linear_logistic_regression_parameters(self):
        self.tolerance_layout = QVBoxLayout()
        self.tolerance_slider_layout = QHBoxLayout()
        self.tolerance_slider_label = QLabel(f"Tolerance")
        self.tolerance_slider = QSlider(Qt.Orientation.Horizontal)
        self.tolerance_slider.setRange(0, 9)
        self.tolerance_slider.valueChanged.connect(self.on_model_parameter_change)
        self.tolerance_value_label = QLabel(f"{self.tolerance()}")

        self.current_parameters = [{"label": self.tolerance_slider_label, "parameter": self.tolerance_slider,
                                    "parameter_layout": self.tolerance_layout,
                                    "parameter_value_layout": self.tolerance_slider_layout,
                                    "parameter_value_label": self.tolerance_value_label,
                                    "parameter_value": self.tolerance, "parameter_range": self.tolerance_range(),
                                    "alias": "tol"}]

        self._add_widgets(self.current_parameters)
        self._add_values(self.current_parameters)

    def _add_decision_tree_parameters(self):
        self.max_depth_layout = QVBoxLayout()
        self.max_depth_box_layout = QHBoxLayout()
        self.max_depth_box_label = QLabel(f"Max Depth")
        self.max_depth_box = QComboBox()
        self.max_depth_box.addItems(["3", "5", "10", "20", "None"])
        self.max_depth_box.currentIndexChanged.connect(self.on_model_parameter_change)

        self.min_samples_split_layout = QVBoxLayout()
        self.min_samples_split_slider_layout = QHBoxLayout()
        self.min_samples_split_slider_label = QLabel(f"Min Samples Split")
        self.min_samples_split_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_samples_split_slider.setRange(0, 2)
        self.min_samples_split_slider.valueChanged.connect(self.on_model_parameter_change)
        self.min_samples_split_value_label = QLabel(f"{self.min_samples_split()}")

        self.min_samples_leaf_layout = QVBoxLayout()
        self.min_samples_leaf_slider_layout = QHBoxLayout()
        self.min_samples_leaf_slider_label = QLabel(f"Min Samples Leaf")
        self.min_samples_leaf_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_samples_leaf_slider.setRange(0, 2)
        self.min_samples_leaf_slider.valueChanged.connect(self.on_model_parameter_change)
        self.min_samples_leaf_value_label = QLabel(f"{self.min_samples_leaf()}")

        self.current_parameters = [{"label": self.max_depth_box_label, "parameter": self.max_depth_box,
                                    "parameter_layout": self.max_depth_layout,
                                    "parameter_value_layout": self.max_depth_box_layout,
                                    "parameter_value": self.max_depth, "parameter_range": None,
                                    "alias": "max_depth"},

                                   {"label": self.min_samples_split_slider_label,
                                    "parameter": self.min_samples_split_slider,
                                    "parameter_layout": self.min_samples_split_layout,
                                    "parameter_value_layout": self.min_samples_split_slider_layout,
                                    "parameter_value_label": self.min_samples_split_value_label,
                                    "parameter_value": self.min_samples_split,
                                    "parameter_range": self.min_samples_split_range(), "alias": "min_samples_split"},

                                   {"label": self.min_samples_leaf_slider_label,
                                    "parameter": self.min_samples_leaf_slider,
                                    "parameter_layout": self.min_samples_leaf_layout,
                                    "parameter_value_layout": self.min_samples_leaf_slider_layout,
                                    "parameter_value_label": self.min_samples_leaf_value_label,
                                    "parameter_value": self.min_samples_leaf,
                                    "parameter_range": self.min_samples_leaf_range(), "alias": "min_samples_leaf"}
                                   ]

        self._add_widgets(self.current_parameters)
        self._add_values(self.current_parameters)

    def _add_random_forest_parameters(self):
        self.n_estimators_layout = QVBoxLayout()
        self.n_estimators_slider_layout = QHBoxLayout()
        self.n_estimators_slider_label = QLabel(f"N Estimators")
        self.n_estimators_slider = QSlider(Qt.Orientation.Horizontal)
        self.n_estimators_slider.setRange(0, 2)
        self.n_estimators_slider.valueChanged.connect(self.on_model_parameter_change)
        self.n_estimators_value_label = QLabel(f"{self.n_estimators()}")

        self.max_depth_layout = QVBoxLayout()
        self.max_depth_box_layout = QHBoxLayout()
        self.max_depth_box_label = QLabel(f"Max Depth")
        self.max_depth_box = QComboBox()
        self.max_depth_box.addItems(["3", "5", "10", "20", "None"])
        self.max_depth_box.currentIndexChanged.connect(self.on_model_parameter_change)

        self.min_samples_split_layout = QVBoxLayout()
        self.min_samples_split_slider_layout = QHBoxLayout()
        self.min_samples_split_slider_label = QLabel(f"Min Samples Split")
        self.min_samples_split_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_samples_split_slider.setRange(0, 2)
        self.min_samples_split_slider.valueChanged.connect(self.on_model_parameter_change)
        self.min_samples_split_value_label = QLabel(f"{self.min_samples_split()}")

        self.min_samples_leaf_layout = QVBoxLayout()
        self.min_samples_leaf_slider_layout = QHBoxLayout()
        self.min_samples_leaf_slider_label = QLabel(f"Min Samples Leaf")
        self.min_samples_leaf_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_samples_leaf_slider.setRange(0, 2)
        self.min_samples_leaf_slider.valueChanged.connect(self.on_model_parameter_change)
        self.min_samples_leaf_value_label = QLabel(f"{self.min_samples_leaf()}")

        self.current_parameters = [{"label": self.n_estimators_slider_label, "parameter": self.n_estimators_slider,
                                    "parameter_layout": self.n_estimators_layout,
                                    "parameter_value_layout": self.n_estimators_slider_layout,
                                    "parameter_value_label": self.n_estimators_value_label,
                                    "parameter_value": self.n_estimators, "parameter_range": self.n_estimators_range(),
                                    "alias": "n_estimators"},

                                   {"label": self.max_depth_box_label, "parameter": self.max_depth_box,
                                    "parameter_layout": self.max_depth_layout,
                                    "parameter_value_layout": self.max_depth_box_layout,
                                    "parameter_value": self.max_depth, "parameter_range": None,
                                    "alias": "max_depth"},

                                   {"label": self.min_samples_split_slider_label,
                                    "parameter": self.min_samples_split_slider,
                                    "parameter_layout": self.min_samples_split_layout,
                                    "parameter_value_layout": self.min_samples_split_slider_layout,
                                    "parameter_value_label": self.min_samples_split_value_label,
                                    "parameter_value": self.min_samples_split,
                                    "parameter_range": self.min_samples_split_range(), "alias": "min_samples_split"},

                                   {"label": self.min_samples_leaf_slider_label,
                                    "parameter": self.min_samples_leaf_slider,
                                    "parameter_layout": self.min_samples_leaf_layout,
                                    "parameter_value_layout": self.min_samples_leaf_slider_layout,
                                    "parameter_value_label": self.min_samples_leaf_value_label,
                                    "parameter_value": self.min_samples_leaf,
                                    "parameter_range": self.min_samples_leaf_range(), "alias": "min_samples_leaf"}
                                   ]

        self._add_widgets(self.current_parameters)
        self._add_values(self.current_parameters)

    def _add_support_vector_regressor_parameters(self):
        self.c_layout = QVBoxLayout()
        self.c_slider_layout = QHBoxLayout()
        self.c_slider_label = QLabel(f"C")
        self.c_slider = QSlider(Qt.Orientation.Horizontal)
        self.c_slider.setRange(0, 3)
        self.c_slider.valueChanged.connect(self.on_model_parameter_change)
        self.c_value_label = QLabel(f"{self.c()}")

        self.gamma_layout = QVBoxLayout()
        self.gamma_box_layout = QHBoxLayout()
        self.gamma_box_label = QLabel(f"Gamma")
        self.gamma_box = QComboBox()
        self.gamma_box.addItems(["scale", "auto", "0.001", "0.01", "0.1", "1"])
        self.gamma_box.currentIndexChanged.connect(self.on_model_parameter_change)

        self.epsilon_layout = QVBoxLayout()
        self.epsilon_slider_layout = QHBoxLayout()
        self.epsilon_slider_label = QLabel(f"Epsilon")
        self.epsilon_slider = QSlider(Qt.Orientation.Horizontal)
        self.epsilon_slider.setRange(0, 4)
        self.epsilon_slider.valueChanged.connect(self.on_model_parameter_change)
        self.epsilon_value_label = QLabel(f"{self.epsilon()}")

        self.current_parameters = [{"label": self.c_slider_label,
                                    "parameter": self.c_slider,
                                    "parameter_layout": self.c_layout,
                                    "parameter_value_layout": self.c_slider_layout,
                                    "parameter_value_label": self.c_value_label,
                                    "parameter_value": self.c,
                                    "parameter_range": self.c_range(),
                                    "alias": "C"},

                                   {"label": self.gamma_box_label,
                                    "parameter": self.gamma_box,
                                    "parameter_layout": self.gamma_layout,
                                    "parameter_value_layout": self.gamma_box_layout,
                                    "parameter_value": self.gamma,
                                    "parameter_range": None,
                                    "alias": "gamma"},

                                   {"label": self.epsilon_slider_label,
                                    "parameter": self.epsilon_slider,
                                    "parameter_layout": self.epsilon_layout,
                                    "parameter_value_layout": self.epsilon_slider_layout,
                                    "parameter_value_label": self.epsilon_value_label,
                                    "parameter_value": self.epsilon,
                                    "parameter_range": self.epsilon_range(),
                                    "alias": "epsilon"}
                                   ]

        self._add_widgets(self.current_parameters)
        self._add_values(self.current_parameters)

    def _add_k_neighbours_parameters(self):
        self.n_neighbors_layout = QVBoxLayout()
        self.n_neighbors_slider_layout = QHBoxLayout()
        self.n_neighbors_slider_label = QLabel(f"N Neighbors")
        self.n_neighbors_slider = QSlider(Qt.Orientation.Horizontal)
        self.n_neighbors_slider.setRange(0, 6)
        self.n_neighbors_slider.valueChanged.connect(self.on_model_parameter_change)
        self.n_neighbors_value_label = QLabel(f"{self.n_neighbors()}")

        self.weights_layout = QVBoxLayout()
        self.weights_box_layout = QHBoxLayout()
        self.weights_box_label = QLabel(f"Weight")
        self.weights_box = QComboBox()
        self.weights_box.addItems(["uniform", "distance"])
        self.weights_box.currentIndexChanged.connect(self.on_model_parameter_change)

        self.metric_layout = QVBoxLayout()
        self.metric_box_layout = QHBoxLayout()
        self.metric_box_label = QLabel(f"Metric")
        self.metric_box = QComboBox()
        self.metric_box.addItems(["euclidean", "manhattan", "minkowski"])
        self.metric_box.currentIndexChanged.connect(self.on_model_parameter_change)

        self.current_parameters = [{"label": self.n_neighbors_slider_label,
                                    "parameter": self.n_neighbors_slider,
                                    "parameter_layout": self.n_neighbors_layout,
                                    "parameter_value_layout": self.n_neighbors_slider_layout,
                                    "parameter_value_label": self.n_neighbors_value_label,
                                    "parameter_value": self.n_neighbors,
                                    "parameter_range": self.n_neighbors_range(),
                                    "alias": "n_neighbors"},

                                   {"label": self.weights_box_label,
                                    "parameter": self.weights_box,
                                    "parameter_layout": self.weights_layout,
                                    "parameter_value_layout": self.weights_box_layout,
                                    "parameter_value": self.weights,
                                    "parameter_range": None,
                                    "alias": "weights"},

                                   {"label": self.metric_box_label,
                                    "parameter": self.metric_box,
                                    "parameter_layout": self.metric_layout,
                                    "parameter_value_layout": self.metric_box_layout,
                                    "parameter_value": self.metric,
                                    "parameter_range": None,
                                    "alias": "metric"},
                                   ]

        self._add_widgets(self.current_parameters)
        self._add_values(self.current_parameters)

    def _add_custom_neural_network_regressor_parameters(self):
        self.lr_layout = QVBoxLayout()
        self.lr_slider_layout = QHBoxLayout()
        self.lr_slider_label = QLabel(f"Learning Rate")
        self.lr_slider = QSlider(Qt.Orientation.Horizontal)
        self.lr_slider.setRange(0, 15)
        self.lr_slider.valueChanged.connect(self.on_model_parameter_change)
        self.lr_value_label = QLabel(f"{self.lr()}")

        self.loss_function_layout = QVBoxLayout()
        self.loss_function_box_layout = QHBoxLayout()
        self.loss_function_box_label = QLabel(f"Loss Function")
        self.loss_function_box = QComboBox()
        self.loss_function_box.addItems(["Mean Squared Error", "Mean Absolute Error", "Huber Loss"])
        self.loss_function_box.currentIndexChanged.connect(self.on_model_parameter_change)

        self.optimizer_layout = QVBoxLayout()
        self.optimizer_box_layout = QHBoxLayout()
        self.optimizer_box_label = QLabel(f"Optimizer")
        self.optimizer_box = QComboBox()
        self.optimizer_box.addItems(["Adam", "SGD", "RMSprop"])
        self.optimizer_box.currentIndexChanged.connect(self.on_model_parameter_change)

        self.current_parameters = [{"label": self.lr_slider_label,
                                    "parameter": self.lr_slider,
                                    "parameter_layout": self.lr_layout,
                                    "parameter_value_layout": self.lr_slider_layout,
                                    "parameter_value_label": self.lr_value_label,
                                    "parameter_value": self.lr,
                                    "parameter_range": self.lr_range(),
                                    "alias": "lr"},

                                   {"label": self.loss_function_box_label,
                                    "parameter": self.loss_function_box,
                                    "parameter_layout": self.loss_function_layout,
                                    "parameter_value_layout": self.loss_function_box_layout,
                                    "parameter_value": self.loss_function,
                                    "parameter_range": None,
                                    "alias": "loss"},

                                   {"label": self.optimizer_box_label,
                                    "parameter": self.optimizer_box,
                                    "parameter_layout": self.optimizer_layout,
                                    "parameter_value_layout": self.optimizer_box_layout,
                                    "parameter_value": self.optimizer,
                                    "parameter_range": None,
                                    "alias": "optimizer"},
                                   ]

        self._add_widgets(self.current_parameters)
        self._add_values(self.current_parameters)

    def _add_support_vector_classifier_parameters(self):
        self.c_layout = QVBoxLayout()
        self.c_slider_layout = QHBoxLayout()
        self.c_slider_label = QLabel(f"C")
        self.c_slider = QSlider(Qt.Orientation.Horizontal)
        self.c_slider.setRange(0, 3)
        self.c_slider.valueChanged.connect(self.on_model_parameter_change)
        self.c_value_label = QLabel(f"{self.c()}")

        self.kernel_layout = QVBoxLayout()
        self.kernel_box_layout = QHBoxLayout()
        self.kernel_box_label = QLabel(f"Kernel")
        self.kernel_box = QComboBox()
        self.kernel_box.addItems(["rbf", "linear"])
        self.kernel_box.currentIndexChanged.connect(self.on_model_parameter_change)

        self.gamma_layout = QVBoxLayout()
        self.gamma_box_layout = QHBoxLayout()
        self.gamma_box_label = QLabel(f"Gamma")
        self.gamma_box = QComboBox()
        self.gamma_box.addItems(["scale", "auto", "0.001", "0.01", "0.1", "1"])
        self.gamma_box.currentIndexChanged.connect(self.on_model_parameter_change)

        self.current_parameters = [{"label": self.c_slider_label,
                                    "parameter": self.c_slider,
                                    "parameter_layout": self.c_layout,
                                    "parameter_value_layout": self.c_slider_layout,
                                    "parameter_value_label": self.c_value_label,
                                    "parameter_value": self.c,
                                    "parameter_range": self.c_range(),
                                    "alias": "C"},

                                   {"label": self.kernel_box_label,
                                    "parameter": self.kernel_box,
                                    "parameter_layout": self.kernel_layout,
                                    "parameter_value_layout": self.kernel_box_layout,
                                    "parameter_value": self.kernel,
                                    "parameter_range": None,
                                    "alias": "gamma"},

                                   {"label": self.gamma_box_label,
                                    "parameter": self.gamma_box,
                                    "parameter_layout": self.gamma_layout,
                                    "parameter_value_layout": self.gamma_box_layout,
                                    "parameter_value": self.gamma,
                                    "parameter_range": None,
                                    "alias": "gamma"}
                                   ]

        self._add_widgets(self.current_parameters)
        self._add_values(self.current_parameters)

    def _add_custom_neural_network_classifier_parameters(self):
        self.lr_layout = QVBoxLayout()
        self.lr_slider_layout = QHBoxLayout()
        self.lr_slider_label = QLabel(f"Learning Rate")
        self.lr_slider = QSlider(Qt.Orientation.Horizontal)
        self.lr_slider.setRange(0, 15)
        self.lr_slider.valueChanged.connect(self.on_model_parameter_change)
        self.lr_value_label = QLabel(f"{self.lr()}")

        self.loss_function_layout = QVBoxLayout()
        self.loss_function_box_layout = QHBoxLayout()
        self.loss_function_box_label = QLabel(f"Loss Function")
        self.loss_function_box = QComboBox()
        self.loss_function_box.addItems(["Binary Cross Entropy"])
        self.loss_function_box.currentIndexChanged.connect(self.on_model_parameter_change)

        self.optimizer_layout = QVBoxLayout()
        self.optimizer_box_layout = QHBoxLayout()
        self.optimizer_box_label = QLabel(f"Optimizer")
        self.optimizer_box = QComboBox()
        self.optimizer_box.addItems(["Adam", "SGD", "RMSprop"])
        self.optimizer_box.currentIndexChanged.connect(self.on_model_parameter_change)

        self.current_parameters = [{"label": self.lr_slider_label,
                                    "parameter": self.lr_slider,
                                    "parameter_layout": self.lr_layout,
                                    "parameter_value_layout": self.lr_slider_layout,
                                    "parameter_value_label": self.lr_value_label,
                                    "parameter_value": self.lr,
                                    "parameter_range": self.lr_range(),
                                    "alias": "lr"},

                                   {"label": self.loss_function_box_label,
                                    "parameter": self.loss_function_box,
                                    "parameter_layout": self.loss_function_layout,
                                    "parameter_value_layout": self.loss_function_box_layout,
                                    "parameter_value": self.loss_function,
                                    "parameter_range": None,
                                    "alias": "loss"},

                                   {"label": self.optimizer_box_label,
                                    "parameter": self.optimizer_box,
                                    "parameter_layout": self.optimizer_layout,
                                    "parameter_value_layout": self.optimizer_box_layout,
                                    "parameter_value": self.optimizer,
                                    "parameter_range": None,
                                    "alias": "optimizer"},
                                   ]

        self._add_widgets(self.current_parameters)
        self._add_values(self.current_parameters)

    def on_model_change(self):
        self._clear_layout()

        if self.model.currentText() == "Linear Regression":
            self._add_linear_logistic_regression_parameters()
        elif self.model.currentText() == "Decision Tree Regressor":
            self._add_decision_tree_parameters()
        elif self.model.currentText() == "Random Forest Regressor":
            self._add_random_forest_parameters()
        elif self.model.currentText() == "Support Vector Regressor":
            self._add_support_vector_regressor_parameters()
        elif self.model.currentText() == "KNeighbors Regressor":
            self._add_k_neighbours_parameters()
        elif self.model.currentText() == "Custom Neural Network Regressor":
            self._add_custom_neural_network_regressor_parameters()

        elif self.model.currentText() == "Logistic Regression":
            self._add_linear_logistic_regression_parameters()
        elif self.model.currentText() == "Decision Tree Classifier":
            self._add_decision_tree_parameters()
        elif self.model.currentText() == "Random Forest Classifier":
            self._add_random_forest_parameters()
        elif self.model.currentText() == "Support Vector Classifier":
            self._add_support_vector_classifier_parameters()
        elif self.model.currentText() == "KNeighbors Classifier":
            self._add_k_neighbours_parameters()
        elif self.model.currentText() == "Custom Neural Network Classifier":
            self._add_custom_neural_network_classifier_parameters()

        self.model_scoring_selection.blockSignals(True)
        self.model_scoring_selection.setCurrentIndex(0)
        self.model_scoring_selection.blockSignals(False)

        self.run_manager.load_model(self.model.currentText(), **self._get_param_dict())

    @staticmethod
    def _check_value(value):
        if value == "None":
            return None
        elif isinstance(value, str) and value.isdigit():
            return int(value)
        else:
            return value

    @staticmethod
    def tolerance_range():
        return np.round(np.linspace(0.0001, 0.000001, 10), 6)

    def tolerance(self):
        return self.tolerance_range()[self.tolerance_slider.value()]

    def max_depth(self):
        return self._check_value(self.max_depth_box.currentText())

    @staticmethod
    def min_samples_split_range():
        return [2, 5, 10, 20]

    def min_samples_split(self):
        return self.min_samples_split_range()[self.min_samples_split_slider.value()]

    @staticmethod
    def min_samples_leaf_range():
        return [1, 2, 5, 10]

    def min_samples_leaf(self):
        return self.min_samples_leaf_range()[self.min_samples_leaf_slider.value()]

    @staticmethod
    def n_estimators_range():
        return [100, 300, 500]

    def n_estimators(self):
        return self.n_estimators_range()[self.n_estimators_slider.value()]

    @staticmethod
    def c_range():
        return [0.1, 1, 10, 100, 1000]

    def c(self):
        return self.c_range()[self.c_slider.value()]

    def kernel(self):
        return self.kernel_box.currentText()

    def gamma(self):
        return self._check_value(self.gamma_box.currentText())

    @staticmethod
    def epsilon_range():
        return [0.01, 0.05, 0.1, 0.5, 1]

    def epsilon(self):
        return self.epsilon_range()[self.epsilon_slider.value()]

    @staticmethod
    def n_neighbors_range():
        return [3, 5, 7, 9, 11, 15, 21]

    def n_neighbors(self):
        return self.n_neighbors_range()[self.n_neighbors_slider.value()]

    def weights(self):
        return self.weights_box.currentText()

    def metric(self):
        return self.metric_box.currentText()

    @staticmethod
    def lr_range():
        start = 1
        end = 4
        return np.round(np.logspace(-start, -end, num=(end - start) * 5 + 1), 6)

    def lr(self):
        return self.lr_range()[self.lr_slider.value()]

    def loss_function(self):
        return self.loss_function_box.currentText()

    def optimizer(self):
        return self.optimizer_box.currentText()

    def _update_parameter_labels(self):
        for parameter in self.current_parameters:
            parameter_value_label = parameter.get("parameter_value_label")
            if parameter_value_label is not None:
                parameter_value_label.setText(f"{parameter["parameter_value"]()}")

    def on_model_parameter_change(self):

        self._update_parameter_labels()

        self.run_manager.load_model(self.model.currentText(), **self._get_param_dict())

        self._set_ui(False)

        self.run_manager.start()

    def on_model_parameter_selection_change(self):
        parameters = {parameter["alias"]: parameter["parameter_range"] for parameter in self.current_parameters}

        param_name = self.model_parameter_selection.currentText()
        scoring = self.model_scoring_selection.currentText()

        if param_name != "Select Parameter" and scoring != "Select Scoring":
            self.run_manager.load_parameters(param_name=param_name,
                                             param_range=parameters[param_name],
                                             scoring=scoring)

            self.run_manager.start()

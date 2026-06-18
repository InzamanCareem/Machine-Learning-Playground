from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotsPanel:
    def __init__(self, canvas_tabs):
        self.canvas_tabs = canvas_tabs

        self.actual_vs_predicted_tab = QWidget()
        self.actual_vs_predicted_tab_layout = QVBoxLayout()
        self.actual_vs_predicted_plot = Figure()
        self.actual_vs_predicted_plot_canvas = FigureCanvas(self.actual_vs_predicted_plot)

        self.residuals_vs_fitted_tab = QWidget()
        self.residuals_vs_fitted_tab_layout = QVBoxLayout()
        self.residuals_vs_fitted_plot = Figure()
        self.residuals_vs_fitted_plot_canvas = FigureCanvas(self.residuals_vs_fitted_plot)

        self.confusion_matrix_tab = QWidget()
        self.confusion_matrix_tab_layout = QVBoxLayout()
        self.confusion_matrix_plot = Figure()
        self.confusion_matrix_plot_canvas = FigureCanvas(self.confusion_matrix_plot)

        self.roc_curve_tab = QWidget()
        self.roc_curve_tab_layout = QVBoxLayout()
        self.roc_curve_plot = Figure()
        self.roc_curve_plot_canvas = FigureCanvas(self.roc_curve_plot)

        self.precision_vs_recall_tab = QWidget()
        self.precision_vs_recall_tab_layout = QVBoxLayout()
        self.precision_vs_recall_plot = Figure()
        self.precision_vs_recall_plot_canvas = FigureCanvas(self.precision_vs_recall_plot)

        self.learning_curve_tab = QWidget()
        self.learning_curve_tab_layout = QVBoxLayout()
        self.learning_curve_plot = Figure()
        self.learning_curve_plot_canvas = FigureCanvas(self.learning_curve_plot)

        self.validation_curve_tab = QWidget()
        self.validation_curve_tab_layout = QVBoxLayout()
        self.validation_curve_plot = Figure()
        self.validation_curve_plot_canvas = FigureCanvas(self.validation_curve_plot)

        self._set_tab()

    def _set_tab(self):
        self.actual_vs_predicted_tab_layout.addWidget(self.actual_vs_predicted_plot_canvas)
        self.actual_vs_predicted_tab.setLayout(self.actual_vs_predicted_tab_layout)

        self.residuals_vs_fitted_tab_layout.addWidget(self.residuals_vs_fitted_plot_canvas)
        self.residuals_vs_fitted_tab.setLayout(self.residuals_vs_fitted_tab_layout)

        self.confusion_matrix_tab_layout.addWidget(self.confusion_matrix_plot_canvas)
        self.confusion_matrix_tab.setLayout(self.confusion_matrix_tab_layout)

        self.roc_curve_tab_layout.addWidget(self.roc_curve_plot_canvas)
        self.roc_curve_tab.setLayout(self.roc_curve_tab_layout)

        self.precision_vs_recall_tab_layout.addWidget(self.precision_vs_recall_plot_canvas)
        self.precision_vs_recall_tab.setLayout(self.precision_vs_recall_tab_layout)

        self.learning_curve_tab_layout.addWidget(self.learning_curve_plot_canvas)
        self.learning_curve_tab.setLayout(self.learning_curve_tab_layout)

        self.validation_curve_tab_layout.addWidget(self.validation_curve_plot_canvas)
        self.validation_curve_tab.setLayout(self.validation_curve_tab_layout)

        self.canvas_tabs.addTab(self.actual_vs_predicted_tab, "Actual vs Predicted")
        self.canvas_tabs.addTab(self.residuals_vs_fitted_tab, "Residuals vs Fitted")
        self.canvas_tabs.addTab(self.confusion_matrix_tab, "Confusion Matrix")
        self.canvas_tabs.addTab(self.roc_curve_tab, "ROC Curve")
        self.canvas_tabs.addTab(self.precision_vs_recall_tab, "Precision vs Recall")
        self.canvas_tabs.addTab(self.learning_curve_tab, "Learning Curve")
        self.canvas_tabs.addTab(self.validation_curve_tab, "Validation Curve")

    def reset_values(self):
        self.actual_vs_predicted_plot.clear()
        self.actual_vs_predicted_plot_canvas.draw()

        self.residuals_vs_fitted_plot.clear()
        self.residuals_vs_fitted_plot_canvas.draw()

    def plot_actual_vs_predicted(self, run):
        self.actual_vs_predicted_plot.clear()
        ax = self.actual_vs_predicted_plot.add_subplot(111)

        ax.scatter(run["actual"], run["predictions"], label="Predictions")

        min_val = min(min(run["actual"]), min(run["predictions"]))
        max_val = max(max(run["actual"]), max(run["predictions"]))

        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="Ideal (y=x)")

        ax.set_title("Actual vs Predicted")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

        ax.legend()
        ax.grid()

        self.actual_vs_predicted_plot_canvas.draw()

    def plot_residuals_vs_fitted(self, run):
        self.residuals_vs_fitted_plot.clear()
        ax = self.residuals_vs_fitted_plot.add_subplot(111)

        residuals = run["actual"] - run["predictions"]

        ax.scatter(run["predictions"], residuals, alpha=0.7)

        ax.axhline(y=0, color="red", linestyle="--")

        ax.set_title("Residuals vs Fitted")
        ax.set_xlabel("Fitted (Predicted)")
        ax.set_ylabel("Residuals")

        ax.grid()

        self.residuals_vs_fitted_plot_canvas.draw()

    def plot_confusion_matrix(self, run):
        self.confusion_matrix_plot.clear()
        ax = self.confusion_matrix_plot.add_subplot(111)

        cm = run["cm"]

        im = ax.imshow(cm, cmap="Blues")

        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        if "classes" in run:
            classes = run["classes"]
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        self.confusion_matrix_plot.colorbar(im, ax=ax)

        self.confusion_matrix_plot_canvas.draw()

    def plot_roc_curve(self, run):
        self.roc_curve_plot.clear()
        ax = self.roc_curve_plot.add_subplot(111)

        ax.plot(run["fpr"], run["tpr"], label=f"AUC = {run["roc_auc"]:.3f}")

        # Random classifier baseline
        ax.plot([0, 1], [0, 1], linestyle="--")

        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        ax.legend()
        ax.grid()

        self.roc_curve_plot_canvas.draw()

    #
    # def plot_precision_vs_recall(self):
    #     self.precision_vs_recall_plot.clear()
    #     ax = self.precision_vs_recall_plot.add_subplot(111)
    #
    #     ax.set_title()
    #     ax.set_xlabel()
    #     ax.set_ylabel()
    #
    #     ax.legend()
    #     ax.grid()
    #
    #     self.precision_vs_recall_plot_canvas.draw()
    #
    #
    # def plot_learning_curve(self):
    #     self.learning_curve_plot.clear()
    #     ax = self.learning_curve_plot.add_subplot(111)
    #
    #     ax.set_title()
    #     ax.set_xlabel()
    #     ax.set_ylabel()
    #
    #     ax.legend()
    #     ax.grid()
    #
    #     self.learning_curve_plot_canvas.draw()
    #
    #
    # def plot_validation_curve(self):
    #     self.validation_curve_plot.clear()
    #     ax = self.validation_curve_plot.add_subplot(111)
    #
    #     ax.set_title()
    #     ax.set_xlabel()
    #     ax.set_ylabel()
    #
    #     ax.legend()
    #     ax.grid()
    #
    #     self.validation_curve_plot_canvas.draw()

    # def plot_curve(self, run):
    #     self.plot.clear()
    #     ax = self.plot.add_subplot(111)
    #
    #     # ax.scatter(run["x"], run["y"], label="Actual", c="red")
    #     ax.scatter(run["y"], run["predictions"], label="Predicted", c="blue")
    #
    #     ax.set_title(run["title"])
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #
    #     ax.legend()
    #     ax.grid()
    #
    #     self.plot_canvas.draw()
    #
    # def compare_plot_curve(self, selected, current):
    #     self.compare_plot.clear()
    #     ax = self.compare_plot.add_subplot(111)
    #
    #     ax.plot(current["x"], current["y"], label="Current")
    #     ax.plot(selected["x"], selected["y"], label="Selected", linestyle="--")
    #
    #     ax.set_title(current["title"])
    #     ax.set_xlabel("Epochs")
    #     ax.set_ylabel("Curve")
    #
    #     ax.legend()
    #     ax.grid()
    #
    #     self.compare_plot_canvas.draw()

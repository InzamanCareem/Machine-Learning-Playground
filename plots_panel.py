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

        self.loss_curve_tab = QWidget()
        self.loss_curve_tab_layout = QVBoxLayout()
        self.loss_curve_plot = Figure()
        self.loss_curve_plot_canvas = FigureCanvas(self.loss_curve_plot)

        self.accuracy_curve_tab = QWidget()
        self.accuracy_curve_tab_layout = QVBoxLayout()
        self.accuracy_curve_plot = Figure()
        self.accuracy_curve_plot_canvas = FigureCanvas(self.accuracy_curve_plot)

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

        self.loss_curve_tab_layout.addWidget(self.loss_curve_plot_canvas)
        self.loss_curve_tab.setLayout(self.loss_curve_tab_layout)

        self.accuracy_curve_tab_layout.addWidget(self.accuracy_curve_plot_canvas)
        self.accuracy_curve_tab.setLayout(self.accuracy_curve_tab_layout)

        self.canvas_tabs.addTab(self.actual_vs_predicted_tab, "Actual vs Predicted")
        self.canvas_tabs.addTab(self.residuals_vs_fitted_tab, "Residuals vs Fitted")
        self.canvas_tabs.addTab(self.confusion_matrix_tab, "Confusion Matrix")
        self.canvas_tabs.addTab(self.roc_curve_tab, "ROC Curve")
        self.canvas_tabs.addTab(self.precision_vs_recall_tab, "Precision vs Recall")
        self.canvas_tabs.addTab(self.learning_curve_tab, "Learning Curve")
        self.canvas_tabs.addTab(self.validation_curve_tab, "Validation Curve")
        self.canvas_tabs.addTab(self.loss_curve_tab, "Loss Curve")
        self.canvas_tabs.addTab(self.accuracy_curve_tab, "Accuracy Curve")

    def reset_values(self):
        self.actual_vs_predicted_plot.clear()
        self.actual_vs_predicted_plot_canvas.draw()

        self.residuals_vs_fitted_plot.clear()
        self.residuals_vs_fitted_plot_canvas.draw()

        self.confusion_matrix_plot.clear()
        self.confusion_matrix_plot_canvas.draw()

        self.roc_curve_plot.clear()
        self.roc_curve_plot_canvas.draw()

        self.precision_vs_recall_plot.clear()
        self.precision_vs_recall_plot_canvas.draw()

        self.learning_curve_plot.clear()
        self.learning_curve_plot_canvas.draw()

        self.validation_curve_plot.clear()
        self.validation_curve_plot_canvas.draw()

        self.loss_curve_plot.clear()
        self.loss_curve_plot_canvas.draw()

        self.accuracy_curve_plot.clear()
        self.accuracy_curve_plot_canvas.draw()

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

        ax.plot([0, 1], [0, 1], linestyle="--")

        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        ax.legend()
        ax.grid()

        self.roc_curve_plot_canvas.draw()

    def plot_precision_vs_recall(self, run):
        self.precision_vs_recall_plot.clear()
        ax = self.precision_vs_recall_plot.add_subplot(111)

        baseline = sum(run["y_true"]) / len(run["y_true"])

        ax.plot(run["recall"], run["precision"], label=f"PR curve (AP={run["ap"]:.3f})")
        ax.fill_between(run["recall"], run["precision"], alpha=0.2)

        ax.axhline(baseline, linestyle="--", label="Baseline")

        ax.set_title("Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        ax.legend()
        ax.grid()

        self.precision_vs_recall_plot_canvas.draw()

    def plot_learning_curve(self, run):
        self.learning_curve_plot.clear()
        ax = self.learning_curve_plot.add_subplot(111)

        ax.plot(run["lc_train_sizes"], run["lc_train_mean"], label="Training score")
        ax.plot(run["lc_train_sizes"], run["lc_val_mean"], label="Validation score")

        ax.set_title("Learning Curve")
        ax.set_xlabel("Training set size")
        ax.set_ylabel("Score")

        ax.legend()
        ax.grid()

        self.learning_curve_plot_canvas.draw()

    def plot_validation_curve(self, run):
        self.validation_curve_plot.clear()
        ax = self.validation_curve_plot.add_subplot(111)

        if run["param_range"] is None:
            return

        ax.plot(run["param_range"], run["vc_train_mean"], label="Training score", marker='o')
        ax.plot(run["param_range"], run["vc_test_mean"], label="Cross-validation score", marker='o')

        ax.set_title("Validation Curve")
        ax.set_xlabel(run["param_name"])
        ax.set_ylabel(run["scoring"])

        ax.legend()
        ax.grid()

        self.validation_curve_plot_canvas.draw()

    def plot_loss_curve(self, run):
        self.loss_curve_plot.clear()
        ax = self.loss_curve_plot.add_subplot(111)

        ax.plot(run["epochs"], run["train_loss"], label="Training loss")
        ax.plot(run["epochs"], run["test_loss"], label="Test loss")

        ax.set_title("Loss Curve")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")

        ax.legend()
        ax.grid()

        self.loss_curve_plot_canvas.draw()

    def plot_accuracy_curve(self, run):
        self.accuracy_curve_plot.clear()
        ax = self.accuracy_curve_plot.add_subplot(111)

        ax.plot(run["epochs"], run["train_acc"], label="Training accuracy")
        ax.plot(run["epochs"], run["test_acc"], label="Test accuracy")

        ax.set_title("Loss Curve")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")

        ax.legend()
        ax.grid()

        self.accuracy_curve_plot_canvas.draw()

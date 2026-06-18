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

        self._set_tab()

    def _set_tab(self):
        self.actual_vs_predicted_tab_layout.addWidget(self.actual_vs_predicted_plot_canvas)
        self.actual_vs_predicted_tab.setLayout(self.actual_vs_predicted_tab_layout)

        self.residuals_vs_fitted_tab_layout.addWidget(self.residuals_vs_fitted_plot_canvas)
        self.residuals_vs_fitted_tab.setLayout(self.residuals_vs_fitted_tab_layout)

        self.canvas_tabs.addTab(self.actual_vs_predicted_tab, "Actual vs Predicted")
        self.canvas_tabs.addTab(self.residuals_vs_fitted_tab, "Residuals vs Fitted")

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

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotsPanel:
    def __init__(self, canvas_tabs):
        self.canvas_tabs = canvas_tabs

        self.first_tab = QWidget()
        self.first_tab_layout = QVBoxLayout()

        self.plot = Figure()
        self.plot_canvas = FigureCanvas(self.plot)

        self.compare_plot = Figure()
        self.compare_plot_canvas = FigureCanvas(self.compare_plot)

        self._set_tab()

    def _set_tab(self):
        self.first_tab_layout.addWidget(self.plot_canvas)
        self.first_tab_layout.addWidget(self.compare_plot_canvas)
        self.first_tab.setLayout(self.first_tab_layout)

        self.canvas_tabs.addTab(self.first_tab, "First Tab")

    def reset_values(self):
        self.plot.clear()
        self.plot_canvas.draw()

        self.compare_plot.clear()
        self.compare_plot_canvas.draw()

    def plot_curve(self, run):
        self.plot.clear()
        ax = self.plot.add_subplot(111)

        ax.scatter(run["x"], run["y"], label="Actual", c="red")
        ax.scatter(run["x"], run["predictions"], label="Predicted", c="blue")

        ax.set_title(run["title"])
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.legend()
        ax.grid()

        self.plot_canvas.draw()

    def compare_plot_curve(self, selected, current):
        self.compare_plot.clear()
        ax = self.compare_plot.add_subplot(111)

        ax.plot(current["x"], current["y"], label="Current")
        ax.plot(selected["x"], selected["y"], label="Selected", linestyle="--")

        ax.set_title(current["title"])
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Curve")

        ax.legend()
        ax.grid()

        self.compare_plot_canvas.draw()

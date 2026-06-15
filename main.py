import sys
from PyQt6.QtWidgets import QApplication
from plot_window import PlotWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlotWindow()
    window.showMaximized()
    sys.exit(app.exec())

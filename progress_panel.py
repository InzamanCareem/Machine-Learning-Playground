from PyQt6.QtWidgets import QProgressBar, QLabel, QComboBox, QVBoxLayout


class ProgressPanel:
    def __init__(self):
        self.progress_layout = QVBoxLayout()

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)

        self.progress_layout.addWidget(QLabel("Progress"))
        self.progress_layout.addWidget(self.progress)

        self.compare_box = QComboBox()
        self.compare_box.addItem("Select run")

        self.progress_layout.addWidget(QLabel("Compare Runs"))
        self.progress_layout.addWidget(self.compare_box)

    def update_dropdown(self):
        self.compare_box.clear()
        self.compare_box.addItem("Select run")

        for i, r in enumerate(self.history):
            self.compare_box.addItem(f"Run {i + 1}: {r['name']}")

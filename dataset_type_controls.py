from PyQt6.QtWidgets import QLabel, QComboBox


class DatasetTypeControls:
    def __init__(self, layout, progress_panel, reset_ui):
        self.layout = layout
        self.progress_panel = progress_panel
        self.reset_ui = reset_ui

        self.dataset = QComboBox()
        self.dataset.addItems(["Regression", "Classification"])
        self.dataset.currentIndexChanged.connect(self.on_dataset_type_change)

        self._set_layout()

    def _set_layout(self):
        self.layout.addWidget(QLabel("Dataset Type"))
        self.layout.addWidget(self.dataset)

    def on_dataset_type_change(self):
        self.progress_panel.set_history([])
        self.progress_panel.set_current_run(None)

        self.reset_ui(self.dataset.currentText())

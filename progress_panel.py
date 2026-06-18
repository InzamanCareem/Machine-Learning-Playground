from PyQt6.QtWidgets import QProgressBar, QLabel, QComboBox


class ProgressPanel:
    def __init__(self, layout, plot_panel):
        self.layout = layout
        self.plot_panel = plot_panel

        self.history = []
        self.current_run = None

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.compare_box = QComboBox()
        self.compare_box.addItem("Select run")

        self.compare_box.currentIndexChanged.connect(self.on_compare_box_change)

    def set_layout(self):
        self.layout.addWidget(QLabel("Progress"))
        self.layout.addWidget(self.progress)

        self.layout.addWidget(QLabel("Compare Runs"))
        self.layout.addWidget(self.compare_box)

    def set_history(self, history):
        self.history = history
        self._update_history()

    def set_current_run(self, current_run):
        self.current_run = current_run

    def save_run(self, new_run):
        if self.current_run is not None:
            self.history.append(self.current_run)

            if len(self.history) > 3:
                self.history.pop(0)

        self.current_run = new_run

        self._update_history()

    def _update_history(self):
        self.compare_box.clear()
        self.compare_box.addItem("Select run")

        for i, r in enumerate(self.history):
            self.compare_box.addItem(f"Run {i + 1}: {r['name']}")

    def on_compare_box_change(self, index):
        run_index = index - 1
        if run_index < 0 or run_index >= len(self.history):
            return

        selected = self.history[run_index]
        current = self.current_run

        self.plot_panel.compare_plot_curve(selected, current)

    def reset_progress_value(self):
        self.progress.setValue(0)

    def reset_values(self):
        self.set_history([])
        self.set_current_run(None)

        self.progress.setValue(0)

        self.compare_box.clear()
        self.compare_box.addItem("Select run")

from training_controller import TrainingController
from progress_panel import ProgressPanel


class RunManager:
    def __init__(self, dataset, model, plots):
        self.history = []
        self.current_run = None

        self.pp = ProgressPanel()
        self.tc = TrainingController(dataset, model, self.history, self.current_run)

        self.plots = plots

    def run(self):
        self.tc.auto_train()

        self.history = self.tc.get_latest_history()
        self.pp.update_dropdown(self.history)

        self.current_run = self.tc.get_latest_run()

        print(self.current_run)
        if self.current_run is not None:
            if self.current_run["model_type"] == "dlr":
                self.pp.compare_box.currentIndexChanged.connect(self.plots.compare_loss_curves)
                self.plots.plot_loss_curve(self.current_run)
            elif self.current_run["model_type"] == "dlc":
                self.pp.compare_box.currentIndexChanged.connect(self.plots.compare_loss_curves)
                self.pp.compare_box.currentIndexChanged.connect(self.plots.compare_accuracy_curves)
                self.plots.plot_loss_curve(self.current_run)
                self.plots.plot_accuracy_curve(self.current_run)
            else:
                self.plots.plot_learning_curve(self.current_run)
                self.plots.plot_validation_curve(self.current_run)

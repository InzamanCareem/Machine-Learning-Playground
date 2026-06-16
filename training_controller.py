from train_worker import TrainWorker


class TrainingController:
    def __init__(self, dataset, model, history, current_run):
        self.dataset = dataset
        self.model = model

        self.history = history
        self.current_run = current_run

        self.dataset_controls = self.dataset.get_controls()
        self.model_controls = self.model.get_controls()

    def set_ui(self, state):
        self.dataset.samples_slider.setEnabled(state)
        self.dataset.features_slider.setEnabled(state)

        self.model.max_depth_slider.setEnabled(state)
        self.model.min_samples_split_slider.setEnabled(state)
        self.model.min_samples_leaf_slider.setEnabled(state)

    def _run_training(self):
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()

        self.set_ui(False)
        # self.progress.setValue(0)

        if self.model.get_model().currentText() == "Custom Neural Network":
            pass
            # self.worker = TrainWorker.from_dl_model(self.dataset.get_dataset().currentText(), self.samples(),
            #                                         self.features(), self.model.get_model().currentText(),
            #                                         self.lr_values[self.lr_slider.value()],
            #                                         self.loss_box.currentText(), self.opt_box.currentText())

        else:
            self.worker = TrainWorker.from_ml_model(self.dataset.get_dataset().currentText(), self.dataset.samples(),
                                                    self.dataset.features(), self.model.get_model().currentText())

        # self.worker.progress.connect(self.progress.setValue)
        self.worker.run_config.connect(self._save_run)
        self.worker.start()

    def get_latest_run(self):
        return self.current_run

    def get_latest_history(self):
        return self.history

    def _save_run(self, run_config):
        if self.current_run is not None:
            self.history.append(self.current_run)

            if len(self.history) > 3:
                self.history.pop(0)

        self.current_run = run_config

        self.set_ui(True)

    def auto_train(self):
        # TODO: add training config to file and use that, do not train every time

        self.dataset_controls[0].valueChanged.connect(self._run_training)
        self.dataset_controls[1].valueChanged.connect(self._run_training)

        self.model_controls[0].valueChanged.connect(self._run_training)
        self.model_controls[1].valueChanged.connect(self._run_training)
        self.model_controls[2].valueChanged.connect(self._run_training)

        # self.lr_slider.valueChanged.connect(self._run_training)
        # self.loss_box.currentIndexChanged.connect(self._run_training)
        # self.opt_box.currentIndexChanged.connect(self._run_training)

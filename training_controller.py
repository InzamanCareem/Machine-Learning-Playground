from train_worker import TrainWorker


class TrainingController:
    def run_training(self):
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()

        self.set_ui(False)
        self.progress.setValue(0)

        if self.model.currentText() == "Custom Neural Network":
            self.worker = TrainWorker.from_dl_model(self.dataset.currentText(), self.samples(), self.features(),
                                                    self.model.currentText(), self.lr_values[self.lr_slider.value()],
                                                    self.loss_box.currentText(), self.opt_box.currentText())
        else:
            self.worker = TrainWorker.from_ml_model(self.dataset.currentText(), self.samples(), self.features(),
                                                    self.model.currentText())

        self.worker.progress.connect(self.progress.setValue)
        self.worker.run_config.connect(self.save_run)
        self.worker.start()

    def save_run(self, run_config):
        if self.current_run is not None:
            self.history.append(self.current_run)

            if len(self.history) > 3:
                self.history.pop(0)

        self.current_run = run_config

        self.update_dropdown()

        if run_config["model_type"] == "dlr":
            self.compare_box.currentIndexChanged.connect(self.compare_loss_curves)
            self.plot_loss_curve(run_config)
        elif run_config["model_type"] == "dlc":
            self.compare_box.currentIndexChanged.connect(self.compare_loss_curves)
            self.compare_box.currentIndexChanged.connect(self.compare_accuracy_curves)
            self.plot_loss_curve(run_config)
            self.plot_accuracy_curve(run_config)
        else:
            self.plot_learning_curve(run_config)
            self.plot_validation_curve(run_config)

        self.set_ui(True)

    def auto_train(self):
        # TODO: add training config to file and use that, do not train every time
        self.samples_slider.valueChanged.connect(self.run_training)
        self.features_slider.valueChanged.connect(self.run_training)

        self.max_depth_slider.valueChanged.connect(self.run_training)
        self.min_samples_split_slider.valueChanged.connect(self.run_training)
        self.min_samples_leaf_slider.valueChanged.connect(self.run_training)

        self.lr_slider.valueChanged.connect(self.run_training)
        self.loss_box.currentIndexChanged.connect(self.run_training)
        self.opt_box.currentIndexChanged.connect(self.run_training)

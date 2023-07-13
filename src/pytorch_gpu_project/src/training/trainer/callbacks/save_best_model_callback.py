import csv
import datetime as dt
import os

import numpy as np
import yaml
from pytorch_accelerated.callbacks import TrainerCallback

from src.training.config.train_config import TrainConfig
from src.utils.model_manager import ModelManager


class SaveBestModelCallback(TrainerCallback):
    """
    A callback which saves the best model during a training run, according to a given metric.
    The best model weights are loaded at the end of the training run.
    """
    def __init__(
        self,
        train_config: TrainConfig,
        watch_metric="eval_loss_epoch",
        greater_is_better: bool = False,
        reset_on_train: bool = True,
        save_optimizer: bool = True,
    ):
        """

        :param save_path: The path to save the checkpoint to. This should end in ``.pt``.
        :param watch_metric: the metric used to compare model performance. This should be accessible from the trainer's run history.    # noqa: E501
        :param greater_is_better: whether an increase in the ``watch_metric`` should be interpreted as the model performing better.  # noqa: E501
        :param reset_on_train: whether to reset the best metric on subsequent training runs. If ``True``, only the metrics observed during the current training run will be compared.  # noqa: E501
        :param save_optimizer: whether to also save the optimizer as part of the model checkpoint
        """
        self.train_config = train_config
        self.watch_metric = watch_metric
        self.greater_is_better = greater_is_better
        self.operator = np.greater if self.greater_is_better else np.less
        self.best_metric = None
        self.best_metric_epoch = None
        self.reset_on_train = reset_on_train
        self.save_optimizer = save_optimizer

    def on_training_run_start(self, args, **kwargs):
        if self.reset_on_train:
            self.best_metric = None

    def on_training_run_epoch_end(self, trainer, **kwargs):
        current_metric = trainer.run_history.get_latest_metric(self.watch_metric)
        if self.best_metric is None:
            self.best_metric = current_metric
            self.best_metric_epoch = trainer.run_history.current_epoch
            self._save_run_config(trainer)
        else:
            is_improvement = self.operator(current_metric, self.best_metric)
            if is_improvement:
                self._save_run_config(trainer)

    def on_training_run_end(self, trainer, **kwargs):
        self._save_run_args_to_yaml(trainer)
        self._save_train_history(trainer)

    def _create_paths(self):
        model_manager = ModelManager()

        time_tag = dt.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        train_run_path = f"{model_manager.base_output_path}/training/{self.train_config.model_name}/{time_tag}"
        if not os.path.exists(train_run_path):
            print("Creating training run path")
            os.makedirs(train_run_path)

        else:
            print("Training run path already exists")

        state_dict_path = f"{train_run_path}/best_model.pt"
        args_path = f"{train_run_path}/args.yaml"
        train_history_path = f"{train_run_path}/train_history.csv"

        return state_dict_path, args_path, train_history_path

    def _save_run_args_to_yaml(self, trainer):
        state_dict_path, args_path, train_history_path = self._create_paths()

        with open(args_path, "w") as f:
            yaml.dump(trainer.run_config.__dict__, f, default_flow_style=False)

    def _save_train_history(self, trainer):
        state_dict_path, args_path, train_history_path = self._create_paths()

        metric_names = trainer.run_history.get_metric_names()
        first_row = ["epoch"] + list(metric_names)

        # format history
        metrics = np.array([trainer.run_history.get_metric_values(m_name) for m_name in metric_names])
        epochs = np.arange(1, len(metrics[0]) + 1)
        hist = np.concatenate(([epochs], metrics), axis=0)
        hist = np.concatenate(([first_row], hist.T), axis=0)

        with open(train_history_path, "w") as f:
            for epoch_row in hist:
                csv_writer = csv.writer(f)
                csv_writer.writerow(epoch_row)

    def _save_run_config(self, trainer):
        state_dict_path, args_path, train_history_path = self._create_paths()

        trainer.save_checkpoint(
            save_path=state_dict_path,
            checkpoint_kwargs={self.watch_metric: self.best_metric},
            save_optimizer=self.save_optimizer,
        )


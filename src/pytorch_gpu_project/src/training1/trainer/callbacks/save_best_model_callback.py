import datetime as dt
import os

import numpy as np
from pytorch_accelerated.callbacks import TrainerCallback

from src.training1.config.train_config import TrainConfig


class SaveBestModelCallback(TrainerCallback):
    """
    A callback which saves the best model during a training run, according to a given metric.
    The best model weights are loaded at the end of the training run.
    """
    BASE_PATH = "src/output/train"

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

        # check that base folder exists
        if not os.path.exists(self.BASE_PATH):
            os.makedirs(self.BASE_PATH)

    def on_training_run_start(self, args, **kwargs):
        if self.reset_on_train:
            self.best_metric = None

    def on_training_run_epoch_end(self, trainer, **kwargs):
        current_metric = trainer.run_history.get_latest_metric(self.watch_metric)
        if self.best_metric is None:
            self.best_metric = current_metric
            self.best_metric_epoch = trainer.run_history.current_epoch
            self._save_run(trainer)
        else:
            is_improvement = self.operator(current_metric, self.best_metric)

            if is_improvement:
                self._save_run(trainer)

    def on_training_run_end(self, trainer, **kwargs):
        trainer.print(
            f"Loading checkpoint with {self.watch_metric}: {self.best_metric}",
            f"from epoch {self.best_metric_epoch}"
        )
        trainer.load_checkpoint(self.save_path)

    def _save_run(self, trainer):
        # create folder for this run
        time_tag = dt.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        self.save_folder_path = f"{self.BASE_PATH}/{time_tag}{self.train_config.model_name}"
        os.mkdir(self.save_folder_path)

        # create paths
        self.state_dict_path = f"{self.save_folder_path}/best_model.pt"
        self.args_path = f"{self.save_folder_path}/args.yaml"
        self.train_history_path = f"{self.save_folder_path}/train_history.csv"

        # save state dicts
        trainer.save_checkpoint(
            save_path=self.state_dict_path,
            checkpoint_kwargs={self.watch_metric: self.best_metric},
            save_optimizer=self.save_optimizer,
        )

        # save train args
        self.train_config.save_to_yaml(self.args_path)

        # TODO: save train history

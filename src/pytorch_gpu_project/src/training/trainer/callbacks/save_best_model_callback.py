import numpy as np
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

        self.model_manager = ModelManager()
        self.run_path = self.model_manager.create_run_path_training(self.train_config.model_name)
        self.state_dict_path = f"{self.run_path}/best_model.pt"
        self.args_path = f"{self.run_path}/run_args.yaml"
        self.train_history_path = f"{self.run_path}/train_history.csv"

    def on_training_run_start(self, args, **kwargs):
        if self.reset_on_train:
            self.best_metric = None

    def on_training_run_epoch_end(self, trainer, **kwargs):
        current_metric = trainer.run_history.get_latest_metric(self.watch_metric)

        if self.best_metric is None:
            self.best_metric = current_metric
            self.best_metric_epoch = trainer.run_history.current_epoch
            trainer.save_checkpoint(
                save_path=self.state_dict_path,
                checkpoint_kwargs={self.watch_metric: self.best_metric},
                save_optimizer=self.save_optimizer,
            )
        else:
            is_improvement = self.operator(current_metric, self.best_metric)
            if is_improvement:
                trainer.save_checkpoint(
                    save_path=self.state_dict_path,
                    checkpoint_kwargs={self.watch_metric: self.best_metric},
                    save_optimizer=self.save_optimizer,
                )

    def on_training_run_end(self, trainer, **kwargs):
        trainer.save_run_history(self.train_history_path)
        self.train_config.save_to_yaml(self.args_path)




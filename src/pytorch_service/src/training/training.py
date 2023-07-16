import timm
import timm.data
import torch.nn as nn
from pytorch_accelerated.trainer import DEFAULT_CALLBACKS
from timm.optim import create_optimizer_v2, optimizer_kwargs

# from src.data.datasets.coin_data import CoinData
from pytorch_service.src.data.datasets.coin_data import CoinData
from pytorch_service.src.training.config.train_config import TrainConfig
from pytorch_service.src.training.trainer.callbacks.save_best_model_callback import (
    SaveBestModelCallback,
)
from pytorch_service.src.training.trainer.timm_trainer import TimmMixupTrainer


class Training:
    def __init__(
        self,
        coin_data: CoinData,
        train_config: TrainConfig,
    ):
        # Setup config and data
        self.train_config = train_config
        self.coin_data = coin_data

        # Setup model
        self.model = self._generate_model()

        # Load data config associated with the model to use in data augmentation pipeline
        self.data_config = timm.data.resolve_data_config(
            vars(self.train_config),
            model=self.model,
        )

        # create datasets
        self.train_dataset, self.val_dataset = self._generate_train_val_datasets()

        # Setup optimizer
        self.optimizer = self._generate_optimizer()

        # Setup trainer
        self.trainer = self._generate_trainer()

        print("Training initialized.")

    def _generate_model(self):
        """Create timm model."""
        in_chans = 3
        if self.train_config.in_chans is not None:
            in_chans = self.train_config.in_chans

        model = timm.create_model(
            self.train_config.model_name,
            pretrained=self.train_config.pretrained,
            in_chans=in_chans,
            num_classes=self.coin_data.num_classes,
            drop_rate=self.train_config.drop_rate,
            drop_path_rate=self.train_config.drop_path_rate,
            drop_block_rate=self.train_config.drop_block_rate,
            global_pool=self.train_config.global_pool,
            bn_momentum=self.train_config.bn_momentum,
            bn_eps=self.train_config.bn_eps,
            scriptable=self.train_config.scriptable,
            checkpoint_path=self.train_config.checkpoint_path,
        )

        return model

    def _generate_train_val_datasets(self):
        """Create train and validation datasets."""
        train_dataset, val_dataset = self.coin_data.generate_train_val_datasets(
            val_pct=0.3,
            image_size=self.data_config["input_size"],
            data_mean=self.data_config["mean"],
            data_std=self.data_config["std"],
            shuffle=True,
        )

        return train_dataset, val_dataset

    def _generate_optimizer(self):
        """Create optimizer."""
        optimizer = create_optimizer_v2(
            self.model,
            **optimizer_kwargs(cfg=self.train_config),
        )
        return optimizer

    def _generate_loss_funcs(self):
        """Set loss functions for train and validation."""
        # TODO: make more variable
        train_loss_fn = nn.CrossEntropyLoss()
        validate_loss_fn = nn.CrossEntropyLoss()

        return train_loss_fn, validate_loss_fn

    def _generate_trainer(self):
        """Create trainer."""
        train_loss_fn, validate_loss_fn = self._generate_loss_funcs()

        trainer = TimmMixupTrainer(
            model=self.model,
            optimizer=self.optimizer,
            loss_func=train_loss_fn,
            eval_loss_fn=validate_loss_fn,
            mixup_args={
                "mixup_alpha": self.train_config.mixup_alpha,
                "cutmix_alpha": self.train_config.cutmix_alpha,
                "label_smoothing": self.train_config.smoothing,
                "num_classes": self.coin_data.num_classes,
            },
            num_classes=self.coin_data.num_classes,
            callbacks=[
                *DEFAULT_CALLBACKS,
                SaveBestModelCallback(
                    watch_metric="accuracy",
                    greater_is_better=True,
                    train_config=self.train_config,
                ),
            ],
        )

        return trainer

    def run(self):
        """Train the model."""
        self.trainer.train(
            per_device_batch_size=self.train_config.batch_size,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            num_epochs=self.train_config.num_epochs,
            create_scheduler_fn=self.trainer.create_scheduler,
        )

        best_run_metrics = self.trainer.get_best_run_metrics()

        run_output = {
            "train_config": dict(self.train_config),
            "best_run_metrics": best_run_metrics
        }
        return run_output

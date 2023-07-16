import csv

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from pytorch_accelerated.trainer import Trainer
from timm.data import Mixup
from timm.scheduler import CosineLRScheduler
from timm.utils import ModelEmaV2


class TimmMixupTrainer(Trainer):
    def __init__(self, eval_loss_fn, mixup_args, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_loss_fn = eval_loss_fn
        self.num_updates = None
        self.mixup_fn = Mixup(**mixup_args)

        self.accuracy = torchmetrics.Accuracy('MULTICLASS', num_classes=num_classes)
        self.ema_accuracy = torchmetrics.Accuracy('MULTICLASS', num_classes=num_classes)
        self.ema_model = None

    def create_scheduler(self):
        # TODO: make this more flexible
        return CosineLRScheduler(
            self.optimizer,
            t_initial=self.run_config.num_epochs,
            cycle_decay=0.5,
            lr_min=1e-6,
            t_in_epochs=True,
            warmup_t=3,
            warmup_lr_init=1e-4,
            cycle_limit=1,
        )

    def training_run_start(self):
        # Model EMA requires the model without a DDP wrapper and before sync batchnorm conversion
        self.ema_model = ModelEmaV2(
            self._accelerator.unwrap_model(self.model), decay=0.9
        )
        if self.run_config.is_distributed:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

    def train_epoch_start(self):
        super().train_epoch_start()
        self.num_updates = self.run_history.current_epoch * len(self._train_dataloader)

    def calculate_train_batch_loss(self, batch):
        xb, yb = batch
        mixup_xb, mixup_yb = self.mixup_fn(xb, yb)
        return super().calculate_train_batch_loss((mixup_xb, mixup_yb))

    def train_epoch_end(
        self,
    ):
        self.ema_model.update(self.model)
        self.ema_model.eval()

        if hasattr(self.optimizer, "sync_lookahead"):
            self.optimizer.sync_lookahead()

    def scheduler_step(self):
        self.num_updates += 1
        if self.scheduler is not None:
            self.scheduler.step_update(num_updates=self.num_updates)

    def calculate_eval_batch_loss(self, batch):
        with torch.no_grad():
            xb, yb = batch
            outputs = self.model(xb)
            val_loss = self.eval_loss_fn(outputs, yb)
            self.accuracy.update(outputs.argmax(-1), yb)

            ema_model_preds = self.ema_model.module(xb).argmax(-1)
            self.ema_accuracy.update(ema_model_preds, yb)

        return {"loss": val_loss, "model_outputs": outputs, "batch_size": xb.size(0)}

    def eval_epoch_end(self):
        super().eval_epoch_end()

        if self.scheduler is not None:
            self.scheduler.step(self.run_history.current_epoch + 1)

        self.run_history.update_metric("accuracy", self.accuracy.compute().cpu())
        self.run_history.update_metric(
            "ema_model_accuracy", self.ema_accuracy.compute().cpu()
        )
        self.accuracy.reset()
        self.ema_accuracy.reset()

    def save_run_history(self, train_history_path):
        metric_names = self.run_history.get_metric_names()
        first_row = ["epoch"] + list(metric_names)

        # format history
        metrics = np.array([self.run_history.get_metric_values(m_name) for m_name in metric_names])
        epochs = np.arange(1, len(metrics[0]) + 1)
        hist = np.concatenate(([epochs], metrics), axis=0)
        hist = np.concatenate(([first_row], hist.T), axis=0)

        # write to csv file
        with open(train_history_path, "w") as f:
            for epoch_row in hist:
                csv_writer = csv.writer(f)
                csv_writer.writerow(epoch_row)
    
    def get_best_run_metrics(self):
        best_metrics = {}
        metric_names = self.run_history.get_metric_names()

        for m_name in metric_names:
            best_metrics[m_name] = max(self.run_history.get_metric_values(m_name))
        
        return best_metrics




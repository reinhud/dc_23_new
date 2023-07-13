from contextlib import suppress
from functools import partial

import confuse
import torch
import torch.nn as nn
import yaml  # type: ignore
from timm import utils
from timm.data import (
    AugMixDataset,
    FastCollateMixup,
    Mixup,
    create_dataset,
    create_loader,
    resolve_data_config,
)
from timm.layers import convert_splitbn_model, convert_sync_batchnorm
from timm.models import (
    create_model,
    load_checkpoint,
    model_parameters,
    resume_checkpoint,
    safe_model_name,
)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler

from src.training.config.training_config import TrainConfig


class TimmTrainerSimple:
    def __init__(
        self,
        train_config: TrainConfig,
    ):
        self.train_config = train_config
        self.train_config.prefetcher = not self.train_config.no_prefetcher

        self.device = self._setup_device()

        self._setup_cuda()

        self.model = self._setup_base_model()

        self.loader_train, self.loader_eval = self._create_loaders()

        self.optimizer = self._setup_optimizer(self.model)

        self.train_loss_fn, self.validate_loss_fn = self._setup_loss()

        self.lr_scheduler, self.num_epochs = self._lr_scheduler()


    def _setup_base_model(self):
        in_chans = 3
        if self.train_config.in_chans is not None:
            in_chans = self.train_config.in_chans
        elif self.train_config.input_size is not None:
            in_chans = self.train_config.input_size[0]

        model = create_model(
            self.train_config.model,
            pretrained=self.train_config.pretrained,
            in_chans=in_chans,
            num_classes=self.train_config.num_classes,
            drop_rate=self.train_config.drop,
            drop_path_rate=self.train_config.drop_path,
            drop_block_rate=self.train_config.drop_block,
            global_pool=self.train_config.gp,
            bn_momentum=self.train_config.bn_momentum,
            bn_eps=self.train_config.bn_eps,
            scriptable=self.train_config.torchscript,
            checkpoint_path=self.train_config.initial_checkpoint,
            # **self.train_config.model_kwargs,      # Not supported atm
        )

        # move model to GPU, enable channels last layout if set
        model.to(device=self.device)
        if self.train_config.channels_last:
            model.to(memory_format=torch.channels_last)

        return model

    def _setup_cuda(self):
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    def _setup_device(self):
        device = utils.init_distributed_device(self.train_config)
        print(f"Using device {device}")
        return device

    def _setup_optimizer(self, model):
        optimizer = create_optimizer_v2(
            model,
            **optimizer_kwargs(cfg=self.train_config),
            **self.train_config.opt_kwargs,
        )
        return optimizer

    def _create_loaders(self):
        # create the train and eval datasets
        dataset_train = create_dataset(
            self.train_config.dataset,
            root=self.train_config.data_dir,
            split=self.train_config.train_split,
            is_training=True,
            class_map=self.train_config.class_map,
            download=self.train_config.dataset_download,
            batch_size=self.train_config.batch_size,
            seed=self.train_config.seed,
            repeats=self.train_config.epoch_repeats,
        )

        dataset_eval = create_dataset(
            self.train_config.dataset,
            root=self.train_config.data_dir,
            split=self.train_config.val_split,
            is_training=False,
            class_map=self.train_config.class_map,
            download=self.train_config.dataset_download,
            batch_size=self.train_config.batch_size,
        )

        # wrap dataset in AugMix helper
        if self.train_config.num_aug_splits > 1:
            dataset_train = AugMixDataset(dataset_train, num_splits=self.train_config.num_aug_splits)

        data_config = resolve_data_config(
            vars(self.train_config),
            model=self.model,
            verbose=utils.is_primary(self.train_config),
        )

        # create data loaders w/ augmentation pipeiine
        train_interpolation = self.train_config.train_interpolation
        if self.train_config.no_aug or not train_interpolation:
            train_interpolation = data_config["interpolation"]
        loader_train = create_loader(
            dataset_train,
            input_size=data_config["input_size"],
            batch_size=self.train_config.batch_size,
            is_training=True,
            use_prefetcher=self.train_config.prefetcher,
            no_aug=self.train_config.no_aug,
            re_prob=self.train_config.reprob,
            re_mode=self.train_config.remode,
            re_count=self.train_config.recount,
            re_split=self.train_config.resplit,
            scale=self.train_config.scale,
            ratio=self.train_config.ratio,
            hflip=self.train_config.hflip,
            vflip=self.train_config.vflip,
            color_jitter=self.train_config.color_jitter,
            auto_augment=self.train_config.aa,
            num_aug_repeats=self.train_config.aug_repeats,
            num_aug_splits=self.train_config.num_aug_splits,
            interpolation=train_interpolation,
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=self.train_config.workers,
            distributed=self.train_config.distributed,
            collate_fn=self.train_config.collate_fn,
            pin_memory=self.train_config.pin_mem,
            device=self.device,
            use_multi_epochs_loader=self.train_config.use_multi_epochs_loader,
            worker_seeding=self.train_config.worker_seeding,
        )

        eval_workers = self.train_config.workers
        if self.train_config.distributed and (
            "tfds" in self.train_config.dataset or "wds" in self.train_config.dataset
        ):
            # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
            eval_workers = min(2, self.train_config.workers)
        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config["input_size"],
            batch_size=self.train_config.validation_batch_size
            or self.train_config.batch_size,
            is_training=False,
            use_prefetcher=self.train_config.prefetcher,
            interpolation=data_config["interpolation"],
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=eval_workers,
            distributed=self.train_config.distributed,
            crop_pct=data_config["crop_pct"],
            pin_memory=self.train_config.pin_mem,
            device=self.device,
        )

        return loader_train, loader_eval

    def _setup_loss(self):
        # TODO: make more variable
        train_loss_fn = nn.CrossEntropyLoss()
        train_loss_fn = train_loss_fn.to(device=self.device)
        validate_loss_fn = nn.CrossEntropyLoss().to(device=self.device)

        return train_loss_fn, validate_loss_fn

    def _lr_scheduler(self):
        updates_per_epoch = (
            len(self.loader_train) + self.train_config.grad_accum_steps - 1
        ) // self.train_config.grad_accum_steps
        lr_scheduler, num_epochs = create_scheduler_v2(
            self.optimizer,
            **scheduler_kwargs(self.train_config),
            updates_per_epoch=updates_per_epoch,
        )

        return lr_scheduler, num_epochs

    def train(self):
        utils.random_seed(self.train_config.seed, self.train_config.rank)
        try:
            for epoch in range(self.num_epochs):
                train_metrics = self._train_one_epoch()
                eval_metrics = self._validate()

                # TODO save metrics and checkpoint

        except KeyboardInterrupt:
            pass

        # print best metrics


    def _train_one_epoch(self):

        self.model.train()

        self.optimizer.zero_grad()

        for batch_idx, (input, target) in enumerate(self.loader_train):

        def _forward():
            input = input.to(self.device)
            target = target.to(self.device)

            output = self.model(input)
            loss = self.train_loss_fn(output, target)

            return loss

        def _backward(_loss):
            _loss.backward()
            self.optimizer.step()

        loss = _forward()
        _backward(loss)

        self.optimizer.zero_grad()

        # TODO: log metrics of this epoch

        # TODO: make step wwith lr scheduler



    def _validate(self):
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(self.loader_eval):
                input = input.to(self.device)
                target = target.to(self.device)

                output = self.model(input)
                if isinstance(output, (tuple, list)):
                        output = output[0]

                loss = self.validate_loss_fn(output, target)



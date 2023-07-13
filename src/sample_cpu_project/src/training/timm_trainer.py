import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torchvision.utils
from timm import utils
from timm.data import (
    AugMixDataset,
    FastCollateMixup,
    Mixup,
    create_dataset,
    create_loader,
    resolve_data_config,
)
from timm.layers import set_fast_norm  # noqa: F401
from timm.layers import convert_splitbn_model, convert_sync_batchnorm
from timm.loss import (
    BinaryCrossEntropy,
    JsdCrossEntropy,
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
)
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
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from src.training.training_config import TrainConfig


class TimmTrainer:
    def __init__(
        self,
        train_config: TrainConfig,
    ):
        self.train_config = train_config

        self.has_apex = self._has_apex()
        self.has_native_amp = self._has_native_amp()
        self.has_wand = self._has_wand()
        self.has_functorch = self._has_functorch()
        self.has_compile = hasattr(torch, "compile")
        self._logger = logging.getLogger("train")

    def _has_apex(self):
        try:
            from apex import amp  # type: ignore # noqa: F401
            from apex.parallel import (  # type: ignore # noqa: F401
                DistributedDataParallel as ApexDDP,
            )
            from apex.parallel import convert_syncbn_model  # type: ignore # noqa: F401

            return True
        except ImportError:
            return False

    def _has_native_amp(self):
        has_native_amp = False
        try:
            if getattr(torch.cuda.amp, "autocast") is not None:
                has_native_amp = True
        except AttributeError:
            pass
        return has_native_amp

    def _has_wand(self):
        try:
            import wandb  # type: ignore # noqa: F401

            return True
        except ImportError:
            return False

    def _has_functorch(self):
        try:
            from functorch.compile import memory_efficient_fusion  # noqa: F401

            return True
        except ImportError:
            return False

    def train(self):
        utils.setup_default_logging()

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        self.train_config.prefetcher = not self.train_config.no_prefetcher
        self.train_config.grad_accum_steps = max(1, self.train_config.grad_accum_steps)
        device = utils.init_distributed_device(self.train_config)
        if self.train_config.distributed:
            self._logger.info(
                "Training in distributed mode with multiple processes, 1 device per process."
                f"Process {self.train_config.rank}, total {self.train_config.world_size},"
                f"device {self.train_config.device}."
            )
        else:
            self._logger.info(
                f"Training with a single process on 1 device ({self.train_config.device})."
            )
        assert self.train_config.rank >= 0

        # resolve AMP arguments based on PyTorch / Apex availability
        use_amp = None
        amp_dtype = torch.float16
        if self.train_config.amp:
            if self.train_config.amp_impl == "apex":
                assert (
                    self.has_apex
                ), "AMP impl specified as APEX but APEX is not installed."
                use_amp = "apex"
                assert self.train_config.amp_dtype == "float16"
            else:
                assert (
                    self.has_native_amp
                ), "Please update PyTorch to a version with native AMP(or use APEX)."
                use_amp = "native"
                assert self.train_config.amp_dtype in ("float16", "bfloat16")
            if self.train_config.amp_dtype == "bfloat16":
                amp_dtype = torch.bfloat16

        utils.random_seed(self.train_config.seed, self.train_config.rank)

        if self.train_config.fuser:
            utils.set_jit_fuser(self.train_config.fuser)
        if self.train_config.fast_norm:
            self.set_fast_norm()

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
            **self.train_config.model_kwargs,
        )
        if self.train_config.head_init_scale is not None:
            with torch.no_grad():
                model.get_classifier().weight.mul_(self.train_config.head_init_scale)
                model.get_classifier().bias.mul_(self.train_config.head_init_scale)
        if self.train_config.head_init_bias is not None:
            nn.init.constant_(
                model.get_classifier().bias, self.train_config.head_init_bias
            )

        if self.train_config.num_classes is None:
            num_class_assert_error_msg = (
                "Model must have `num_classes` attr if not set on cmd line/config."
            )
            assert hasattr(model, "num_classes"), num_class_assert_error_msg
            self.train_config.num_classes = (
                model.num_classes
            )  # FIXME handle model default vs config num_classes more elegantly # noqa: E501

        if self.train_config.grad_checkpointing:
            model.set_grad_checkpointing(enable=True)

        if utils.is_primary(self.train_config):
            self._logger.info(
                f"Model {safe_model_name(self.train_config.model)} created,"
                f"param count:{sum([m.numel() for m in model.parameters()])}"
            )

        data_config = resolve_data_config(
            vars(self.train_config),
            model=model,
            verbose=utils.is_primary(self.train_config),
        )

        # setup augmentation batch splits for contrastive loss or split bn
        num_aug_splits = 0
        if self.train_config.aug_splits > 0:
            assert self.train_config.aug_splits > 1, "A split of 1 makes no sense"
            num_aug_splits = self.train_config.aug_splits

        # enable split bn (separate bn stats per batch-portion)
        if self.train_config.split_bn:
            assert num_aug_splits > 1 or self.train_config.resplit
            model = convert_splitbn_model(model, max(num_aug_splits, 2))

        # move model to GPU, enable channels last layout if set
        model.to(device=device)
        if self.train_config.channels_last:
            model.to(memory_format=torch.channels_last)

        # setup synchronized BatchNorm for distributed training
        if self.train_config.distributed and self.train_config.sync_bn:
            self.train_config.dist_bn = ""  # disable dist_bn when sync BN active
            assert not self.train_config.split_bn
            if self.has_apex and use_amp == "apex":
                # Apex SyncBN used with Apex AMP
                # WARNING this won't currently work with models using BatchNormAct2d
                model = convert_syncbn_model(model)  # type: ignore # noqa: F821
            else:
                model = convert_sync_batchnorm(model)
            if utils.is_primary(self.train_config):
                self._logger.info(
                    "Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using "
                    "zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled."
                )

        if self.train_config.torchscript:
            assert not self.train_config.torchcompile
            assert not use_amp == "apex", "Cannot use APEX AMP with torchscripted model"
            assert (
                not self.train_config.sync_bn
            ), "Cannot use SyncBatchNorm with torchscripted model"
            model = torch.jit.script(model)

        if not self.train_config.lr:
            global_batch_size = (
                self.train_config.batch_size
                * self.train_config.world_size
                * self.train_config.grad_accum_steps
            )
            batch_ratio = global_batch_size / self.train_config.lr_base_size
            if not self.train_config.lr_base_scale:
                on = self.train_config.opt.lower()
                self.train_config.lr_base_scale = (
                    "sqrt" if any([o in on for o in ("ada", "lamb")]) else "linear"
                )
            if self.train_config.lr_base_scale == "sqrt":
                batch_ratio = batch_ratio**0.5
            self.train_config.lr = self.train_config.lr_base * batch_ratio
            if utils.is_primary(self.train_config):
                self._logger.info(
                    f"Learning rate ({self.train_config.lr}) calculated from",
                    f"base learning rate ({self.train_config.lr_base})",
                    f"and effective global batch size ({global_batch_size})",
                    f"with {self.train_config.lr_base_scale} scaling.",
                )

        optimizer = create_optimizer_v2(
            model,
            **optimizer_kwargs(cfg=self.train_config),
            **self.train_config.opt_kwargs,
        )

        # setup automatic mixed-precision (AMP) loss scaling and op casting
        amp_autocast = suppress  # do nothing
        loss_scaler = None
        if use_amp == "apex":
            assert device.type == "cuda"
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")  # type: ignore # noqa: F821
            loss_scaler = ApexScaler()
            if utils.is_primary(self.train_config):
                self._logger.info("Using NVIDIA APEX AMP. Training in mixed precision.")
        elif use_amp == "native":
            try:
                amp_autocast = partial(
                    torch.autocast, device_type=device.type, dtype=amp_dtype
                )
            except (AttributeError, TypeError):
                # fallback to CUDA only AMP for PyTorch < 1.10
                assert device.type == "cuda"
                amp_autocast = torch.cuda.amp.autocast
            if device.type == "cuda" and amp_dtype == torch.float16:
                # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
                loss_scaler = NativeScaler()
            if utils.is_primary(self.train_config):
                self._logger.info(
                    "Using native Torch AMP. Training in mixed precision."
                )
        else:
            if utils.is_primary(self.train_config):
                self._logger.info("AMP not enabled. Training in float32.")

        # optionally resume from a checkpoint
        resume_epoch = None
        if self.train_config.resume:
            resume_epoch = resume_checkpoint(
                model,
                self.train_config.resume,
                optimizer=None if self.train_config.no_resume_opt else optimizer,
                loss_scaler=None if self.train_config.no_resume_opt else loss_scaler,
                log_info=utils.is_primary(self.train_config),
            )

        # setup exponential moving average of model weights, SWA could be used here too
        model_ema = None
        if self.train_config.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
            model_ema = utils.ModelEmaV2(
                model,
                decay=self.train_config.model_ema_decay,
                device="cpu" if self.train_config.model_ema_force_cpu else None,
            )
            if self.train_config.resume:
                load_checkpoint(
                    model_ema.module, self.train_config.resume, use_ema=True
                )

        # setup distributed training
        if self.train_config.distributed:
            if self.has_apex and use_amp == "apex":
                # Apex DDP preferred unless native amp is activated
                if utils.is_primary(self.train_config):
                    self._logger.info("Using NVIDIA APEX DistributedDataParallel.")
                model = ApexDDP(model, delay_allreduce=True)  # type: ignore # noqa: F821
            else:
                if utils.is_primary(self.train_config):
                    self._logger.info("Using native Torch DistributedDataParallel.")
                model = NativeDDP(
                    model,
                    device_ids=[device],
                    broadcast_buffers=not self.train_config.no_ddp_bb,
                )
            # NOTE: EMA model does not need to be wrapped by DDP

        if self.train_config.torchcompile:
            # torch compile should be done after DDP
            torchcompile_assert_error_msg = (("A version of torch w/ torch.compile()"),)
            ("is required for --compile, possibly a nightly.")
            assert self.has_compile, torchcompile_assert_error_msg
            model = torch.compile(model, backend=self.train_config.torchcompile)

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

        # setup mixup / cutmix
        collate_fn = None
        mixup_fn = None
        mixup_active = (
            self.train_config.mixup > 0
            or self.train_config.cutmix > 0.0
            or self.train_config.cutmix_minmax is not None
        )
        if mixup_active:
            mixup_args = dict(
                mixup_alpha=self.train_config.mixup,
                cutmix_alpha=self.train_config.cutmix,
                cutmix_minmax=self.train_config.cutmix_minmax,
                prob=self.train_config.mixup_prob,
                switch_prob=self.train_config.mixup_switch_prob,
                mode=self.train_config.mixup_mode,
                label_smoothing=self.train_config.smoothing,
                num_classes=self.train_config.num_classes,
            )
            if self.train_config.prefetcher:
                assert (
                    not num_aug_splits
                )  # collate conflict (need to support deinterleaving in collate mixup)
                collate_fn = FastCollateMixup(**mixup_args)
            else:
                mixup_fn = Mixup(**mixup_args)

        # wrap dataset in AugMix helper
        if num_aug_splits > 1:
            dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

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
            num_aug_splits=num_aug_splits,
            interpolation=train_interpolation,
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=self.train_config.workers,
            distributed=self.train_config.distributed,
            collate_fn=collate_fn,
            pin_memory=self.train_config.pin_mem,
            device=device,
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
            device=device,
        )

        # setup loss function
        if self.train_config.jsd_loss:
            assert num_aug_splits > 1  # JSD only valid with aug splits set
            train_loss_fn = JsdCrossEntropy(
                num_splits=num_aug_splits, smoothing=self.train_config.smoothing
            )
        elif mixup_active:
            # smoothing is handled with mixup target transform which outputs sparse, soft targets
            if self.train_config.bce_loss:
                train_loss_fn = BinaryCrossEntropy(
                    target_threshold=self.train_config.bce_target_thresh
                )
            else:
                train_loss_fn = SoftTargetCrossEntropy()
        elif self.train_config.smoothing:
            if self.train_config.bce_loss:
                train_loss_fn = BinaryCrossEntropy(
                    smoothing=self.train_config.smoothing,
                    target_threshold=self.train_config.bce_target_thresh,
                )
            else:
                train_loss_fn = LabelSmoothingCrossEntropy(
                    smoothing=self.train_config.smoothing
                )
        else:
            train_loss_fn = nn.CrossEntropyLoss()
        train_loss_fn = train_loss_fn.to(device=device)
        validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

        # setup checkpoint saver and eval metric tracking
        eval_metric = self.train_config.eval_metric
        best_metric = None
        best_epoch = None
        saver = None
        output_dir = None
        if utils.is_primary(self.train_config):
            if self.train_config.experiment:
                exp_name = self.train_config.experiment
            else:
                exp_name = "-".join(
                    [
                        datetime.now().strftime("%Y%m%d-%H%M%S"),
                        safe_model_name(self.train_config.model),
                        str(data_config["input_size"][-1]),
                    ]
                )
            output_dir = utils.get_outdir(
                self.train_config.output
                if self.train_config.output
                else "./output/train",
                exp_name,
            )
            decreasing = True if eval_metric == "loss" else False
            saver = utils.CheckpointSaver(
                model=model,
                optimizer=optimizer,
                args=self.train_config,
                model_ema=model_ema,
                amp_scaler=loss_scaler,
                checkpoint_dir=output_dir,
                recovery_dir=output_dir,
                decreasing=decreasing,
                max_history=self.train_config.checkpoint_hist,
            )
            with open(os.path.join(output_dir, "args.yaml"), "w") as f:
                f.write(self.train_config._get_config_text())

        if utils.is_primary(self.train_config) and self.train_config.log_wandb:
            if self.has_wandb:
                wandb.init(project=self.train_config.experiment, config=self.train_config)  # type: ignore # noqa: F821,E501
            else:
                self._logger.warning(
                    "You've requested to log metrics to wandb but package not found. "
                    "Metrics not being logged to wandb, try `pip install wandb`"
                )

        # setup learning rate schedule and starting epoch
        updates_per_epoch = (
            len(loader_train) + self.train_config.grad_accum_steps - 1
        ) // self.train_config.grad_accum_steps
        lr_scheduler, num_epochs = create_scheduler_v2(
            optimizer,
            **scheduler_kwargs(self.train_config),
            updates_per_epoch=updates_per_epoch,
        )
        start_epoch = 0
        if self.train_config.start_epoch is not None:
            # a specified start_epoch will always override the resume epoch
            start_epoch = self.train_config.start_epoch
        elif resume_epoch is not None:
            start_epoch = resume_epoch
        if lr_scheduler is not None and start_epoch > 0:
            if self.train_config.sched_on_updates:
                lr_scheduler.step_update(start_epoch * updates_per_epoch)
            else:
                lr_scheduler.step(start_epoch)

        if utils.is_primary(self.train_config):
            self._logger.info(
                f"Scheduled epochs: {num_epochs}. LR stepped per"
                '{"epoch" if lr_scheduler.t_in_epochs else "update"}.'
            )

        try:
            for epoch in range(start_epoch, num_epochs):
                if hasattr(dataset_train, "set_epoch"):
                    dataset_train.set_epoch(epoch)
                elif self.train_config.distributed and hasattr(
                    loader_train.sampler, "set_epoch"
                ):
                    loader_train.sampler.set_epoch(epoch)

                train_metrics = self._train_one_epoch(
                    epoch,
                    model,
                    loader_train,
                    optimizer,
                    train_loss_fn,
                    self.train_config,
                    lr_scheduler=lr_scheduler,
                    saver=saver,
                    output_dir=output_dir,
                    amp_autocast=amp_autocast,
                    loss_scaler=loss_scaler,
                    model_ema=model_ema,
                    mixup_fn=mixup_fn,
                )

                if self.train_config.distributed and self.train_config.dist_bn in (
                    "broadcast",
                    "reduce",
                ):
                    if utils.is_primary(self.train_config):
                        self._logger.info(
                            "Distributing BatchNorm running means and vars"
                        )
                    utils.distribute_bn(
                        model,
                        self.train_config.world_size,
                        self.train_config.dist_bn == "reduce",
                    )

                eval_metrics = self._validate(
                    model,
                    loader_eval,
                    validate_loss_fn,
                    self.train_config,
                    amp_autocast=amp_autocast,
                )

                if model_ema is not None and not self.train_config.model_ema_force_cpu:
                    if self.train_config.distributed and self.train_config.dist_bn in (
                        "broadcast",
                        "reduce",
                    ):
                        utils.distribute_bn(
                            model_ema,
                            self.train_config.world_size,
                            self.train_config.dist_bn == "reduce",
                        )

                    ema_eval_metrics = self.validate(
                        model_ema.module,
                        loader_eval,
                        validate_loss_fn,
                        self.train_config,
                        amp_autocast=amp_autocast,
                        log_suffix=" (EMA)",
                    )
                    eval_metrics = ema_eval_metrics

                if output_dir is not None:
                    lrs = [param_group["lr"] for param_group in optimizer.param_groups]
                    utils.update_summary(
                        epoch,
                        train_metrics,
                        eval_metrics,
                        filename=os.path.join(output_dir, "summary.csv"),
                        lr=sum(lrs) / len(lrs),
                        write_header=best_metric is None,
                        log_wandb=self.train_config.log_wandb and self.has_wandb,
                    )

                if saver is not None:
                    # save proper checkpoint with eval metric
                    save_metric = eval_metrics[eval_metric]
                    best_metric, best_epoch = saver.save_checkpoint(
                        epoch, metric=save_metric
                    )

                if lr_scheduler is not None:
                    # step LR for next epoch
                    lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

        except KeyboardInterrupt:
            pass

        if best_metric is not None:
            self._logger.info(
                "*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch)
            )

    def _train_one_epoch(
        self,
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        train_config,
        device=torch.device("cuda"),
        lr_scheduler=None,
        saver=None,
        output_dir=None,
        amp_autocast=suppress,
        loss_scaler=None,
        model_ema=None,
        mixup_fn=None,
    ):
        if train_config.mixup_off_epoch and epoch >= train_config.mixup_off_epoch:
            if train_config.prefetcher and loader.mixup_enabled:
                loader.mixup_enabled = False
            elif mixup_fn is not None:
                mixup_fn.mixup_enabled = False

        second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        has_no_sync = hasattr(model, "no_sync")
        update_time_m = utils.AverageMeter()
        data_time_m = utils.AverageMeter()
        losses_m = utils.AverageMeter()

        model.train()

        accum_steps = train_config.grad_accum_steps
        last_accum_steps = len(loader) % accum_steps
        updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
        num_updates = epoch * updates_per_epoch
        last_batch_idx = len(loader) - 1
        last_batch_idx_to_accum = len(loader) - last_accum_steps

        data_start_time = update_start_time = time.time()
        optimizer.zero_grad()
        update_sample_count = 0
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_batch_idx
            need_update = last_batch or (batch_idx + 1) % accum_steps == 0
            update_idx = batch_idx // accum_steps
            if batch_idx >= last_batch_idx_to_accum:
                accum_steps = last_accum_steps

            if not train_config.prefetcher:
                input, target = input.to(device), target.to(device)
                if mixup_fn is not None:
                    input, target = mixup_fn(input, target)
            if train_config.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # multiply by accum steps to get equivalent for full update
            data_time_m.update(accum_steps * (time.time() - data_start_time))

            def _forward():
                with amp_autocast():
                    output = model(input)
                    loss = loss_fn(output, target)
                if accum_steps > 1:
                    loss /= accum_steps
                return loss

            def _backward(_loss):
                if loss_scaler is not None:
                    loss_scaler(
                        _loss,
                        optimizer,
                        clip_grad=train_config.clip_grad,
                        clip_mode=train_config.clip_mode,
                        parameters=model_parameters(
                            model, exclude_head="agc" in train_config.clip_mode
                        ),
                        create_graph=second_order,
                        need_update=need_update,
                    )
                else:
                    _loss.backward(create_graph=second_order)
                    if need_update:
                        if train_config.clip_grad is not None:
                            utils.dispatch_clip_grad(
                                model_parameters(
                                    model, exclude_head="agc" in train_config.clip_mode
                                ),
                                value=train_config.clip_grad,
                                mode=train_config.clip_mode,
                            )
                        optimizer.step()

            if has_no_sync and not need_update:
                with model.no_sync():
                    loss = _forward()
                    _backward(loss)
            else:
                loss = _forward()
                _backward(loss)

            if not train_config.distributed:
                losses_m.update(loss.item() * accum_steps, input.size(0))
            update_sample_count += input.size(0)

            if not need_update:
                data_start_time = time.time()
                continue

            num_updates += 1
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

            if train_config.synchronize_step and device.type == "cuda":
                torch.cuda.synchronize()
            time_now = time.time()
            update_time_m.update(time.time() - update_start_time)
            update_start_time = time_now

            if update_idx % train_config.log_interval == 0:
                lrl = [param_group["lr"] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                if train_config.distributed:
                    reduced_loss = utils.reduce_tensor(
                        loss.data, train_config.world_size
                    )
                    losses_m.update(reduced_loss.item() * accum_steps, input.size(0))
                    update_sample_count *= train_config.world_size

                if utils.is_primary(train_config):
                    try:
                        dummy = f"{100. * update_idx / (updates_per_epoch - 1):>3.0f}"
                    except ZeroDivisionError:
                        dummy = "NaN"
                    self._logger.info(
                        f"Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} "
                        f"({dummy}%)]  "
                        f"Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  "
                        f"Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  "
                        f"({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  "
                        f"LR: {lr:.3e}  "
                        f"Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})"
                    )

                    if train_config.save_images and output_dir:
                        torchvision.utils.save_image(
                            input,
                            os.path.join(output_dir, "train-batch-%d.jpg" % batch_idx),
                            padding=0,
                            normalize=True,
                        )

            if (
                saver is not None
                and train_config.recovery_interval
                and ((update_idx + 1) % train_config.recovery_interval == 0)
            ):
                saver.save_recovery(epoch, batch_idx=update_idx)

            if lr_scheduler is not None:
                lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

            update_sample_count = 0
            data_start_time = time.time()
            # end for

        if hasattr(optimizer, "sync_lookahead"):
            optimizer.sync_lookahead()

        return OrderedDict([("loss", losses_m.avg)])

    def _validate(
        self,
        model,
        loader,
        loss_fn,
        train_config,
        device=torch.device("cpu"),  # TODO: fix for cpu
        amp_autocast=suppress,
        log_suffix="",
    ):
        batch_time_m = utils.AverageMeter()
        losses_m = utils.AverageMeter()
        top1_m = utils.AverageMeter()
        top5_m = utils.AverageMeter()

        model.eval()

        end = time.time()
        last_idx = len(loader) - 1
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(loader):
                last_batch = batch_idx == last_idx
                if not train_config.prefetcher:
                    input = input.to(device)
                    target = target.to(device)
                if train_config.channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)

                with amp_autocast():
                    output = model(input)
                    if isinstance(output, (tuple, list)):
                        output = output[0]

                    # augmentation reduction
                    reduce_factor = train_config.tta
                    if reduce_factor > 1:
                        output = output.unfold(0, reduce_factor, reduce_factor).mean(
                            dim=2
                        )
                        target = target[0 : target.size(0) : reduce_factor]

                    loss = loss_fn(output, target)
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

                if train_config.distributed:
                    reduced_loss = utils.reduce_tensor(
                        loss.data, train_config.world_size
                    )
                    acc1 = utils.reduce_tensor(acc1, train_config.world_size)
                    acc5 = utils.reduce_tensor(acc5, train_config.world_size)
                else:
                    reduced_loss = loss.data

                if device.type == "cuda":
                    torch.cuda.synchronize()

                losses_m.update(reduced_loss.item(), input.size(0))
                top1_m.update(acc1.item(), output.size(0))
                top5_m.update(acc5.item(), output.size(0))

                batch_time_m.update(time.time() - end)
                end = time.time()
                if utils.is_primary(train_config) and (
                    last_batch or batch_idx % train_config.log_interval == 0
                ):
                    log_name = "Test" + log_suffix
                    self._logger.info(
                        f"{log_name}: [{batch_idx:>4d}/{last_idx}]  "
                        f"Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  "
                        f"Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  "
                        f"Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  "
                        f"Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})"
                    )

        metrics = OrderedDict(
            [("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)]
        )

        return metrics

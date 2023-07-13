import confuse
import yaml  # type: ignore
from timm.data import FastCollateMixup, Mixup


class TrainConfig:
    """Class to hold all training base_configuration parameters."""

    BASE_CONFIG_PATH = "src/training/config/base_config.yaml"

    base_config = confuse.Configuration("Base Train Config")
    base_config.set_file(BASE_CONFIG_PATH)

    # TODO: add type to input from config
    def __init__(
        self,
        data_dir=base_config["data_dir"]["default"].get(),
        dataset=base_config["dataset"]["default"].get(),
        train_split=base_config["train_split"]["default"].get(),
        val_split=base_config["val_split"]["default"].get(),
        dataset_download=base_config["dataset_download"]["default"].get(),
        class_map=base_config["class_map"]["default"].get(),
        model=base_config["model"]["default"].get(),
        pretrained=base_config["pretrained"]["default"].get(),
        initial_checkpoint=base_config["initial_checkpoint"]["default"].get(),
        resume=base_config["resume"]["default"].get(),
        no_resume_opt=base_config["no_resume_opt"]["default"].get(),
        num_classes=base_config["num_classes"]["default"].get(),
        gp=base_config["gp"]["default"].get(),
        img_size=base_config["img_size"]["default"].get(),
        in_chans=base_config["in_chans"]["default"].get(),
        input_size=base_config["input_size"]["default"].get(),
        crop_pct=base_config["crop_pct"]["default"].get(),
        mean=base_config["mean"]["default"].get(),
        std=base_config["std"]["default"].get(),
        interpolation=base_config["interpolation"]["default"].get(),
        batch_size=base_config["batch_size"]["default"].get(),
        validation_batch_size=base_config["validation_batch_size"]["default"].get(),
        channels_last=base_config["channels_last"]["default"].get(),
        fuser=base_config["fuser"]["default"].get(),
        grad_accum_steps=base_config["grad_accum_steps"]["default"].get(),
        grad_checkpointing=base_config["grad_checkpointing"]["default"].get(),
        fast_norm=base_config["fast_norm"]["default"].get(),
        model_kwargs=dict(base_config["model_kwargs"]["default"].get()),
        head_init_scale=base_config["head_init_scale"]["default"].get(),
        head_init_bias=base_config["head_init_bias"]["default"].get(),
        torchscript=base_config["torchscript"]["default"].get(),
        torchcompile=base_config["torchcompile"]["default"].get(),
        opt=base_config["opt"]["default"].get(),
        opt_eps=base_config["opt_eps"]["default"].get(),
        opt_betas=base_config["opt_betas"]["default"].get(),
        momentum=base_config["momentum"]["default"].get(),
        weight_decay=base_config["weight_decay"]["default"].get(),
        clip_grad=base_config["clip_grad"]["default"].get(),
        clip_mode=base_config["clip_mode"]["default"].get(),
        layer_decay=base_config["layer_decay"]["default"].get(),
        opt_kwargs=dict(base_config["opt_kwargs"]["default"].get()),
        sched=base_config["sched"]["default"].get(),
        sched_on_updates=base_config["sched_on_updates"]["default"].get(),
        lr=base_config["lr"]["default"].get(),
        lr_base=base_config["lr_base"]["default"].get(),
        lr_base_size=base_config["lr_base_size"]["default"].get(),
        lr_base_scale=base_config["lr_base_scale"]["default"].get(),
        lr_noise=base_config["lr_noise"]["default"].get(),
        lr_noise_pct=base_config["lr_noise_pct"]["default"].get(),
        lr_noise_std=base_config["lr_noise_std"]["default"].get(),
        lr_cycle_mul=base_config["lr_cycle_mul"]["default"].get(),
        lr_cycle_decay=base_config["lr_cycle_decay"]["default"].get(),
        lr_cycle_limit=base_config["lr_cycle_limit"]["default"].get(),
        lr_k_decay=base_config["lr_k_decay"]["default"].get(),
        warmup_lr=base_config["warmup_lr"]["default"].get(),
        min_lr=base_config["min_lr"]["default"].get(),
        epochs=base_config["epochs"]["default"].get(),
        epoch_repeats=base_config["epoch_repeats"]["default"].get(),
        start_epoch=base_config["start_epoch"]["default"].get(),
        decay_milestones=base_config["decay_milestones"]["default"].get(),
        decay_epochs=base_config["decay_epochs"]["default"].get(),
        warmup_epochs=base_config["warmup_epochs"]["default"].get(),
        warmup_prefix=base_config["warmup_prefix"]["default"].get(),
        cooldown_epochs=base_config["cooldown_epochs"]["default"].get(),
        patience_epochs=base_config["patience_epochs"]["default"].get(),
        decay_rate=base_config["decay_rate"]["default"].get(),
        no_aug=base_config["no_aug"]["default"].get(),
        scale=base_config["scale"]["default"].get(),
        ratio=base_config["ratio"]["default"].get(),
        hflip=base_config["hflip"]["default"].get(),
        vflip=base_config["vflip"]["default"].get(),
        color_jitter=base_config["color_jitter"]["default"].get(),
        aa=base_config["aa"]["default"].get(),
        aug_repeats=base_config["aug_repeats"]["default"].get(),
        aug_splits=base_config["aug_splits"]["default"].get(),
        jsd_loss=base_config["jsd_loss"]["default"].get(),
        bce_loss=base_config["bce_loss"]["default"].get(),
        bce_target_thresh=base_config["bce_target_thresh"]["default"].get(),
        reprob=base_config["reprob"]["default"].get(),
        remode=base_config["remode"]["default"].get(),
        recount=base_config["recount"]["default"].get(),
        resplit=base_config["resplit"]["default"].get(),
        mixup=base_config["mixup"]["default"].get(),
        cutmix=base_config["cutmix"]["default"].get(),
        cutmix_minmax=base_config["cutmix_minmax"]["default"].get(),
        mixup_prob=base_config["mixup_prob"]["default"].get(),
        mixup_switch_prob=base_config["mixup_switch_prob"]["default"].get(),
        mixup_mode=base_config["mixup_mode"]["default"].get(),
        mixup_off_epoch=base_config["mixup_off_epoch"]["default"].get(),
        smoothing=base_config["smoothing"]["default"].get(),
        train_interpolation=base_config["train_interpolation"]["default"].get(),
        drop=base_config["drop"]["default"].get(),
        drop_connect=base_config["drop_connect"]["default"].get(),
        drop_path=base_config["drop_path"]["default"].get(),
        drop_block=base_config["drop_block"]["default"].get(),
        bn_momentum=base_config["bn_momentum"]["default"].get(),
        bn_eps=base_config["bn_eps"]["default"].get(),
        sync_bn=base_config["sync_bn"]["default"].get(),
        dist_bn=base_config["dist_bn"]["default"].get(),
        world_size=base_config["world_size"]["default"].get(),
        split_bn=base_config["split_bn"]["default"].get(),
        model_ema=base_config["model_ema"]["default"].get(),
        model_ema_force_cpu=base_config["model_ema_force_cpu"]["default"].get(),
        model_ema_decay=base_config["model_ema_decay"]["default"].get(),
        seed=base_config["seed"]["default"].get(),
        worker_seeding=base_config["worker_seeding"]["default"].get(),
        log_interval=base_config["log_interval"]["default"].get(),
        recovery_interval=base_config["recovery_interval"]["default"].get(),
        checkpoint_hist=base_config["checkpoint_hist"]["default"].get(),
        workers=base_config["workers"]["default"].get(),
        save_images=base_config["save_images"]["default"].get(),
        amp=base_config["amp"]["default"].get(),
        amp_dtype=base_config["amp_dtype"]["default"].get(),
        amp_impl=base_config["amp_impl"]["default"].get(),
        no_ddp_bb=base_config["no_ddp_bb"]["default"].get(),
        synchronize_step=base_config["synchronize_step"]["default"].get(),
        pin_mem=base_config["pin_mem"]["default"].get(),
        no_prefetcher=base_config["no_prefetcher"]["default"].get(),
        output=base_config["output"]["default"].get(),
        experiment=base_config["experiment"]["default"].get(),
        eval_metric=base_config["eval_metric"]["default"].get(),
        tta=base_config["tta"]["default"].get(),
        use_multi_epochs_loader=base_config["use_multi_epochs_loader"]["default"].get(),
        log_wandb=base_config["log_wandb"]["default"].get(),
        distributed=base_config["distributed"]["default"].get(),
    ):
        # TODO: maybe use auto initializer? https://stackoverflow.com/questions/1389180/automatically-initialize-instance-variables     # noqa: E501
        self.data_dir = data_dir
        self.dataset = dataset
        self.train_split = train_split
        self.val_split = val_split
        self.dataset_download = dataset_download
        self.class_map = class_map
        self.model = model
        self.pretrained = pretrained
        self.initial_checkpoint = initial_checkpoint
        self.resume = resume
        self.no_resume_opt = no_resume_opt
        self.num_classes = num_classes
        self.gp = gp
        self.img_size = img_size
        self.in_chans = in_chans
        self.input_size = input_size
        self.crop_pct = crop_pct
        self.mean = mean
        self.std = std
        self.interpolation = interpolation
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.channels_last = channels_last
        self.fuser = fuser
        self.grad_accum_steps = self._setup_grad_accum_steps(grad_accum_steps)
        self.grad_checkpointing = grad_checkpointing
        self.fast_norm = fast_norm
        self.model_kwargs = model_kwargs
        self.head_init_scale = head_init_scale
        self.head_init_bias = head_init_bias
        self.torchscript = torchscript
        self.torchcompile = torchcompile
        self.opt = opt
        self.opt_eps = opt_eps
        self.opt_betas = opt_betas
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.clip_grad = clip_grad
        self.clip_mode = clip_mode
        self.layer_decay = layer_decay
        self.opt_kwargs = opt_kwargs
        self.sched = sched
        self.sched_on_updates = sched_on_updates
        self.lr_base = lr_base
        self.lr_base_size = lr_base_size
        self.lr_base_scale = lr_base_scale
        self.lr_noise = lr_noise
        self.lr_noise_pct = lr_noise_pct
        self.lr_noise_std = lr_noise_std
        self.lr_cycle_mul = lr_cycle_mul
        self.lr_cycle_decay = lr_cycle_decay
        self.lr_cycle_limit = lr_cycle_limit
        self.lr_k_decay = lr_k_decay
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        self.epochs = epochs
        self.epoch_repeats = epoch_repeats
        self.start_epoch = start_epoch
        self.decay_milestones = decay_milestones
        self.decay_epochs = decay_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_prefix = warmup_prefix
        self.cooldown_epochs = cooldown_epochs
        self.patience_epochs = patience_epochs
        self.decay_rate = decay_rate
        self.no_aug = no_aug
        self.scale = scale
        self.ratio = ratio
        self.hflip = hflip
        self.vflip = vflip
        self.color_jitter = color_jitter
        self.aa = aa
        self.aug_repeats = aug_repeats
        self.aug_splits = aug_splits
        self.jsd_loss = jsd_loss
        self.bce_loss = bce_loss
        self.bce_target_thresh = bce_target_thresh
        self.reprob = reprob
        self.remode = remode
        self.recount = recount
        self.resplit = resplit
        self.mixup = mixup
        self.cutmix = cutmix
        self.cutmix_minmax = cutmix_minmax
        self.mixup_prob = mixup_prob
        self.mixup_switch_prob = mixup_switch_prob
        self.mixup_mode = mixup_mode
        self.mixup_off_epoch = mixup_off_epoch
        self.smoothing = smoothing
        self.train_interpolation = train_interpolation
        self.drop = drop
        self.drop_connect = drop_connect
        self.drop_path = drop_path
        self.drop_block = drop_block
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.sync_bn = sync_bn
        self.dist_bn = dist_bn
        self.world_size = world_size
        self.split_bn = split_bn
        self.model_ema = model_ema
        self.model_ema_force_cpu = model_ema_force_cpu
        self.model_ema_decay = model_ema_decay
        self.seed = seed
        self.worker_seeding = worker_seeding
        self.log_interval = log_interval
        self.recovery_interval = recovery_interval
        self.checkpoint_hist = checkpoint_hist
        self.workers = workers
        self.save_images = save_images
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.amp_impl = amp_impl
        self.no_ddp_bb = no_ddp_bb
        self.synchronize_step = synchronize_step
        self.pin_mem = pin_mem
        self.no_prefetcher = no_prefetcher
        self.output = output
        self.experiment = experiment
        self.eval_metric = eval_metric
        self.tta = tta
        self.use_multi_epochs_loader = use_multi_epochs_loader
        self.log_wandb = log_wandb
        self.distributed=distributed

        self.lr, self.lr_base_scale = self._setup_lr(lr, lr_base_scale)

        self.num_aug_splits = self._num_aug_splits()
        self.collate_fn, self.mixup_fn = self._setup_mixup_cutmix()

    def _setup_grad_accum_steps(self, grad_accum_steps):
        return max(1, grad_accum_steps)

    def _setup_lr(self, lr, lr_base_scale):
        """Resolve conflicts in specified lr args"""
        if lr is None:
            global_batch_size = (
                self.batch_size
                * self.world_size
                * self.grad_accum_steps
            )
            batch_ratio = global_batch_size / self.lr_base_size
            if not lr_base_scale:
                on = self.opt.lower()
                lr_base_scale = (
                    "sqrt" if any([o in on for o in ("ada", "lamb")]) else "linear"
                )
            if lr_base_scale == "sqrt":
                batch_ratio = batch_ratio**0.5
            lr = self.lr_base * batch_ratio

        return lr, lr_base_scale

    def _num_aug_splits(self):
        """Setup augmentation batch splits for contrastive loss or split bn."""
        num_aug_splits = 0
        if self.aug_splits is None or self.aug_splits < 0:
            print("Invalid aug splits {}, using default of 0".format(self.num_aug_splits))
        elif self.aug_splits == 1:
            print("A split of 1 makes no sense. Using default of 0")
        else:
            num_aug_splits = self.aug_splits
        return num_aug_splits

    def _setup_mixup_cutmix(self):
        # setup mixup / cutmix
        collate_fn = None
        mixup_fn = None
        mixup_active = (
            self.mixup > 0
            or self.cutmix > 0.0
            or self.cutmix_minmax is not None
        )
        if mixup_active:
            mixup_args = dict(
                mixup_alpha=self.mixup,
                cutmix_alpha=self.cutmix,
                cutmix_minmax=self.cutmix_minmax,
                prob=self.mixup_prob,
                switch_prob=self.mixup_switch_prob,
                mode=self.mixup_mode,
                label_smoothing=self.smoothing,
                num_classes=self.num_classes,
            )
            if self.prefetcher:
                assert (
                    not self.num_aug_splits
                )  # collate conflict (need to support deinterleaving in collate mixup)
                collate_fn = FastCollateMixup(**mixup_args)
            else:
                mixup_fn = Mixup(**mixup_args)

        return collate_fn, mixup_fn

    def _get_config_text(self) -> str:
        """Return a string containing the base_config in yaml format."""
        return yaml.safe_dump(self.__dict__, default_flow_style=False)

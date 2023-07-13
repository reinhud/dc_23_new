import confuse
import yaml


class TrainConfig:
    """Class to hold all training base_configuration parameters."""

    BASE_CONFIG_PATH = "src/training1/config/base_config.yaml"

    base_config = confuse.Configuration("Base Train Config")
    base_config.set_file(BASE_CONFIG_PATH)

    def __init__(
        self,
        batch_size: int = base_config["batch_size"]["default"].get(),
        num_epochs: int = base_config["num_epochs"]["default"].get(),
        model_name: str = base_config["model_name"]["default"].get(),
        pretrained: bool = base_config["pretrained"]["default"].get(),
        num_classes: int = base_config["num_classes"]["default"].get(),
        in_chans: int = base_config["in_chans"]["default"].get(),
        drop_rate: float = base_config["drop_rate"]["default"].get(),
        drop_path_rate: float = base_config["drop_path_rate"]["default"].get(),
        drop_block_rate: float = base_config["drop_block_rate"]["default"].get(),
        global_pool: str = base_config["global_pool"]["default"].get(),
        bn_momentum: float = base_config["bn_momentum"]["default"].get(),
        bn_eps: float = base_config["bn_eps"]["default"].get(),
        scriptable: bool = base_config["scriptable"]["default"].get(),
        checkpoint_path: str = base_config["checkpoint_path"]["default"].get(),
        mixup_alpha: float = base_config["mixup_alpha"]["default"].get(),
        mixup_prob: float = base_config["mixup_prob"]["default"].get(),
        cutmix_alpha: float = base_config["cutmix_alpha"]["default"].get(),
        smoothing: float = base_config["smoothing"]["default"].get(),
        opt: str = base_config["opt"]["default"].get(),
        lr: float = base_config["lr"]["default"].get(),
        weight_decay: float = base_config["weight_decay"]["default"].get(),
        momentum: float = base_config["momentum"]["default"].get(),
        sched: str = base_config["sched"]["default"].get(),
        grad_accum_steps: int = base_config["grad_accum_steps"]["default"].get(),
    ):
        # training parameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        # model parameters
        self.model_name = model_name
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.drop_block_rate = drop_block_rate
        self.global_pool = global_pool
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.scriptable = scriptable
        self.checkpoint_path = checkpoint_path
        # mixup parameters
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_alpha = cutmix_alpha
        self.smoothing = smoothing
        self.opt = opt
        self.lr = self._resolve_lr(lr)
        # optimizer parameters
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.momentum = momentum
        # scheduler parameters
        self.sched = sched
        self.grad_accum_steps = grad_accum_steps

    def _resolve_lr(self, lr):
        if lr is None:
            # TODO: calculate reasonable default lr
            # how do i get the world size dynamically using pytorch accelerate?
            # or how could i use these params with it
            """global_batch_size = (
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
            self.train_config.lr = self.train_config.lr_base * batch_ratio"""
            return 0.01
        else:
            return lr

    def get_mixup_args(self):
        return {
            "mixup_alpha": self.mixup_alpha,
            "mixup_prob": self.mixup_prob,
            "cutmix_alpha": self.cutmix_alpha,
            "label_smoothing": self.smoothing,
            "num_classes": self.num_classes,
        }

    def save_to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

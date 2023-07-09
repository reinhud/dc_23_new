import confuse
import timm
import torch
from timm.loss import BinaryCrossEntropy
from timm.optim import create_optimizer_v2

from src.data.datasets.coin_data import CoinData, CoinDataFolder
from src.training.trainer import Trainer


# TODO: load from env
class TrainingConfig:
    def __init__(self):
        self.config = confuse.Configuration("train_config", __name__)
        self.config.set_file('train_config.yaml')

        self.model = self.config["ModelUsed"].get()
        self.model_timm_name = self.config["Model"][self.model]["TimmName"].get()
        self.input_size = self.config["General"]["InputSize"].get()
        self.num_epochs = self.config["General"]["NumEpochs"].get()
        self.batch_size = self.config["General"]["BatchSize"].get()
        self.optimizer = self.config["ModelUsed"].get()
        self.loss = self.config["ModelUsed"].get()
        self.lr = self.config["ModelUsed"].get()
        self.smoothing = self.config["ModelUsed"].get()
        self.mixup = self.config["ModelUsed"].get()
        self.cutmix = self.config["ModelUsed"].get()
        self.bce_target_thresh = self.config["ModelUsed"].get()


def training():
    train_config = TrainingConfig()
    # load data
    coin_data = CoinData(folder=CoinDataFolder.TYPES_EXAMPLE)
    num_classes = len(coin_data.images_and_targets)

    mixup_args = dict(
        mixup_alpha=train_config.mixup,
        cutmix_alpha=train_config.cutmix,
        label_smoothing=train_config.smoothing,
        num_classes=num_classes,
    )

    # Create model using timm
    model = timm.create_model(
        "resnet50d", pretrained=False, num_classes=num_classes, drop_path_rate=0.05
    )

    # Load data config associated with the model to use in data augmentation pipeline
    data_config = timm.data.resolve_data_config({}, model=model, verbose=True)
    data_mean = data_config["mean"]
    data_std = data_config["std"]

    train_dataset, eval_dataset = coin_data.generate_train_val_datasets(val_pct=0.3, image_size=train_config.image_size, data_mean=data_mean, data_std=data_std)

    # Create optimizer
    optimizer = create_optimizer_v2(
        model, opt="lookahead_AdamW", lr=lr, weight_decay=0.01
    )

    # As we are using Mixup, we can use BCE during training and CE for evaluation
    train_loss_fn = BinaryCrossEntropy(
        target_threshold=train_config.
        bce_target_thresh, smoothing=train_config.smoothing
    )
    validate_loss_fn = torch.nn.CrossEntropyLoss()

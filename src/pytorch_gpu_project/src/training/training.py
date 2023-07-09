import timm
import torch

from src.data.coin_data import CoinData, CoinDataFolder
from src.training.trainer import Trainer


def training():
    # Set training arguments, hardcoded here for clarity
    image_size = (224, 224)
    lr = 5e-3
    smoothing = 0.1
    mixup = 0.2
    cutmix = 1.0
    batch_size = 16
    bce_target_thresh = 0.2
    num_epochs = 50

    # load data
    coin_data = CoinData(folder=CoinDataFolder.TYPES_EXAMPLE)
    num_classes = len(coin_data.images_and_targets)

    mixup_args = dict(
        mixup_alpha=mixup,
        cutmix_alpha=cutmix,
        label_smoothing=smoothing,
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

    train_dataset, eval_dataset = coin_data.generate_train_val_datasets(val_pct=0.3, image_size=image_size, data_mean=data_mean, data_std=data_std)

    # Create optimizer
    optimizer = timm.optim.create_optimizer_v2(
        model, opt="lookahead_AdamW", lr=lr, weight_decay=0.01
    )

    # As we are using Mixup, we can use BCE during training and CE for evaluation
    train_loss_fn = timm.loss.BinaryCrossEntropy(
        target_threshold=bce_target_thresh, smoothing=smoothing
    )
    validate_loss_fn = torch.nn.CrossEntropyLoss()

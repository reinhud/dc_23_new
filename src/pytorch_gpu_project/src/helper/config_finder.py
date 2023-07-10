from dataclasses import dataclass

import confuse
import yaml  # type: ignore


@dataclass
class DefaultTrainConfig:
    """Dataclass for the default training config"""

    data_path: str
    config_path: str
    epochs: int
    batch_size: int


def default_train_config1():
    config = confuse.Configuration(
        "TrainConfigFinder",
    )
    config.set_file("src/training/train_config/base_config.yaml")

    base_data_path = config["default_data_path"].get()
    base_config_path = config["default_train_config_path"].get()

    try:
        with open(base_config_path, "r") as f:
            cfg = yaml.safe_load(f)
            if cfg is None:
                print(
                    f"Specified default config file {base_config_path} is empty",
                    'defaulting to "timm" default config',
                )
                base_config_path = ""
            else:
                return base_config_path
    except Exception:
        base_config_path = ""

    return base_data_path, base_config_path


def get_default_train_config() -> DefaultTrainConfig:
    """Returns the default config for training."""
    config = confuse.Configuration(
        "TrainConfigFinder",
    )
    config.set_file("src/training/train_config/base_config.yaml")

    data_path = config["default_data_path"].get()
    config_path = config["default_train_config_path"].get()
    epochs = config["default_epochs"].get()
    batch_size = config["default_batch_size"].get()

    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            if cfg is None:
                print(
                    f"Specified default config file {config_path} is empty,"
                    'defaulting to "timm" default config.'
                )
                config_path = ""
    except Exception:
        config_path = ""

    return DefaultTrainConfig(
        data_path=data_path,
        config_path=config_path,
        epochs=epochs,
        batch_size=batch_size,
    )

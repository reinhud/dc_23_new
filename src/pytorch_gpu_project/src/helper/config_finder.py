import confuse
import yaml  # TODO: ignore this


def default_train_config():
    config = confuse.Configuration('TrainConfigFinder',)
    config.set_file('src/training/train_config/base_config.yaml')

    base_data_path = config["default_data_path"].get()
    base_config_path = config["default_train_config_path"].get()

    try:
        with open(base_config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            if cfg is None:
                print(f'Specified default config file {base_config_path} is empty, defaulting to "timm" default config')
                base_config_path = ''
            else:
                return base_config_path
    except Exception:
        base_config_path = ''

    return base_data_path, base_config_path

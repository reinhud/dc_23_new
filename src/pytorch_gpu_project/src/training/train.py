import argparse

import confuse

from src.training.training_config import TrainConfig
from training.timm_trainer import TimmTrainer

config = confuse.Configuration('Train Setup Parser')
config.set_file('src/training/train_config/base_config.yaml')


def cli_train_args():
    parser = argparse.ArgumentParser()
    for k in config.keys():
        flags = config[k]["parser"]["flags"].get()
        nargs = config[k]["parser"]["nargs"].get()
        # dtype = config[k]["parser"]["type"].get()
        metavar = config[k]["parser"]["metavar"].get()
        help = config[k]["parser"]["help"].get()
        # action = config[k]["parser"]["action"].get()
        # const = config[k]["parser"]["const"].get()

        """print(dtype)
        print(type(dtype))

        if action == 'utils.ParseKwargs':
            action = None  # TODO: fix this: ValueError: not enough values to unpack (expected 2, got 1)
        if action == "store_true":
            action = None  # TODO: fix this, does it need to be fixed as boolean defults are stored in config anyways?  # noqa: E501

        if dtype is not None:
            dtype = getattr(__builtins__, dtype)"""

        parser.add_argument(
            *flags,
            nargs=nargs[0],
            # type=dtype,
            metavar=metavar,
            help=help,
            dest=k  # f"{k}.default", #this would be nice to override confuse, but defaults are initialized upon import and not after argparse was set up # noqa: E501
            # action=action,
            # const=const
        )
    return parser.parse_args()


if __name__ == '__main__':
    args = cli_train_args()

    # config.set_args(args, dots=True)
    # print(config['data_dir']["default"].get())

    # print(vars(args))

    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    train_config = TrainConfig(**cli_args)

    trainer = TimmTrainer(train_config)

    trainer.train()

    # print(config['opt_kwargs']["default"].get())
    # print(type(dict(config['opt_kwargs']["default"].get())))

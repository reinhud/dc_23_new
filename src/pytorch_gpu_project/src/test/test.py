import argparse

import confuse
from timm import utils

config = confuse.Configuration("MyGreatApp", __name__)
config.set_file("src/test/config.yaml")

parser = argparse.ArgumentParser()


def _setup_cli_args(config, parser):
    for k in config.keys():
        flags = config[k]["parser"]["flags"].get()
        nargs = config[k]["parser"]["nargs"].get()
        dtype = config[k]["parser"]["type"].get()
        metavar = config[k]["parser"]["metavar"].get()
        help = config[k]["parser"]["help"].get()
        action = config[k]["parser"]["action"].get()
        const = config[k]["parser"]["const"].get()

        if action == "utils.ParseKwargs":
            parser.add_argument(
                *flags,
                nargs=nargs[0],
                type=getattr(__builtins__, str(dtype)),
                metavar=metavar,
                help=help,
                dest=f"{k}.default",
                # action=utils.ParseKwargs   # TODO: fix this: ValueError: not enough values to unpack (expected 2, got 1)
            )
        else:
            parser.add_argument(
                *flags,
                nargs=nargs[0],
                type=getattr(__builtins__, str(dtype)),
                metavar=metavar,
                help=help,
                dest=f"{k}.default",
                action=action,
                const=const,
            )

        args = parser.parse_args()

        print(config["data_dir"]["default"].get())
        print("#######")
        print(args)

        return args


if __name__ == "__main__":
    args = _setup_cli_args(config, parser)
    config.set_args(args, dots=True)

    print(config["data_dir"]["default"].get())

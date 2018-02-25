import argparse

from logging import getLogger

import yaml
from moke_config import create_config

from .lib.logger import setup_logger
from .config import Config

logger = getLogger(__name__)

CMD_LIST = ['self', 'opt', 'eval', 'play_gui', 'nboard']


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", help="what to do", choices=CMD_LIST)
    parser.add_argument("-c", help="specify config yaml", dest="config_file")
    parser.add_argument("--new", help="run from new best model", action="store_true")
    parser.add_argument("--type", help="deprecated. Please use -c instead")
    parser.add_argument("--total-step", help="set TrainerConfig.start_total_steps", type=int)
    return parser


def setup(config: Config, args):
    config.opts.new = args.new
    if args.total_step is not None:
        config.trainer.start_total_steps = args.total_step
    config.resource.create_directories()
    setup_logger(config.resource.main_log_path)


def start():
    parser = create_parser()
    args = parser.parse_args()
    if args.type:
        print("I'm very sorry. --type option was deprecated. Please use -c option instead!")
        return 1

    if args.config_file:
        with open(args.config_file, "rt") as f:
            config = create_config(Config, yaml.load(f))
    else:
        config = create_config(Config)
    setup(config, args)

    if args.cmd != "nboard":
        logger.info(f"config type: {config.type}")

    if args.cmd == "self":
        from .worker import self_play
        return self_play.start(config)
    elif args.cmd == 'opt':
        from .worker import optimize
        return optimize.start(config)
    elif args.cmd == 'eval':
        from .worker import evaluate
        return evaluate.start(config)
    elif args.cmd == 'play_gui':
        from .play_game import gui
        return gui.start(config)
    elif args.cmd == 'nboard':
        from .play_game import nboard
        return nboard.start(config)

import argparse

from .lib.logger import setup_logger
from .config import Config

CMD_LIST = ['self', 'opt', 'eval']


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", help="what to do", choices=CMD_LIST)
    parser.add_argument("--new", help="run from new best model", action="store_true")
    return parser


def setup(config: Config, args):
    config.opts.new = args.new
    config.resource.create_directories()
    setup_logger(config.resource.main_log_path)


def start():
    parser = create_parser()
    args = parser.parse_args()

    config = Config()
    setup(config, args)
    if args.cmd == "self":
        from .worker import self_play
        return self_play.start(config)
    elif args.cmd == 'opt':
        from .worker import optimize
        return optimize.start(config)
    elif args.cmd == 'eval':
        from .worker import evaluate
        return evaluate.start(config)

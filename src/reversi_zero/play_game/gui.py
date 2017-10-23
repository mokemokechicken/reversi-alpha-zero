from logging import getLogger

from reversi_zero.config import Config

logger = getLogger(__name__)


def start(config: Config):
    return EvaluateWorker(config).start()

from logging import getLogger

from reversi_zero.config import Config
from reversi_zero.lib.data_helper import get_next_generation_model_dirs

logger = getLogger(__name__)


def start(config: Config):
    return EvaluateWorker(config).start()


class EvaluateWorker:
    def __init__(self, config: Config):
        """

        :param config:
        """
        self.config = config
        self.best_model = None

    def start(self):
        while True:
            ng_model, paths = self.load_next_generation_model()
            ng_is_great = self.evaluate_model(ng_model)
            if ng_is_great:
                self.save_as_new_best_model(ng_model)
            self.remove_model(paths)

    def load_next_generation_model(self):
        config_files, weight_files = get_next_generation_model_dirs(self.config.resource)
        config_path, weight_path = config_files[0], weight_files[0]


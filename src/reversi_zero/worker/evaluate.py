import os
from logging import getLogger
from random import random
from time import sleep

from reversi_zero.agent.model import ReversiModel
from reversi_zero.agent.player import ReversiPlayer
from reversi_zero.config import Config
from reversi_zero.env.reversi_env import ReversiEnv, Player
from reversi_zero.lib.data_helper import get_next_generation_model_dirs
from reversi_zero.lib.model_helpler import save_as_best_model, load_best_model_weight

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
        self.best_model = self.load_best_model()

        while True:
            ng_model, model_dir = self.load_next_generation_model()
            logger.debug(f"start evaluate model {model_dir}")
            ng_is_great = self.evaluate_model(ng_model)
            if ng_is_great:
                logger.debug(f"New Model become best model: {model_dir}")
                save_as_best_model(ng_model)
                self.best_model = ng_model
            self.remove_model(model_dir)

    def evaluate_model(self, ng_model):
        results = []
        for game_idx in range(self.config.eval.game_num):
            # ng_win := if ng_model win -> 1, lose -> 0, draw -> None
            ng_win = self.play_game(self.best_model, ng_model)
            logger.debug(f"game {game_idx}: ng_win={ng_win}")
            if ng_win is not None:
                results.append(ng_win)
        return sum(results) / len(results) >= self.config.eval.replace_rate

    def play_game(self, best_model, ng_model):
        env = ReversiEnv().reset()

        best_player = ReversiPlayer(self.config, best_model, play_config=self.config.eval.play_config)
        ng_player = ReversiPlayer(self.config, ng_model, play_config=self.config.eval.play_config)
        best_is_black = random() < 0.5
        if best_is_black:
            black, white = best_player, ng_player
        else:
            black, white = ng_player, best_player

        observation = env.observation
        while not env.done:
            if env.next_player == Player.black:
                action = black.action(observation.black, observation.white)
            else:
                action = white.action(observation.white, observation.black)
            observation, info = env.step(action)

        if env.winner == Player.black:
            if best_is_black:
                return 0
            else:
                return 1
        elif env.winner == Player.white:
            if best_is_black:
                return 1
            else:
                return 0
        else:
            return None

    def load_best_model(self):
        model = ReversiModel(self.config)
        load_best_model_weight(model)
        return model

    def load_next_generation_model(self):
        rc = self.config.resource
        while True:
            dirs = get_next_generation_model_dirs(self.config.resource)
            if dirs:
                break
            sleep(10)
        model_dir = dirs[0]
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        model = ReversiModel(self.config)
        model.load(config_path, weight_path)
        return model, model_dir

    def remove_model(self, model_dir):
        rc = self.config.resource
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        os.remove(config_path)
        os.remove(weight_path)
        os.rmdir(model_dir)

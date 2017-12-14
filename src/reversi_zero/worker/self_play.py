import os
from datetime import datetime
from logging import getLogger
from random import random
from time import time

from reversi_zero.agent.player import ReversiPlayer
from reversi_zero.config import Config
from reversi_zero.env.reversi_env import Board, Winner
from reversi_zero.env.reversi_env import ReversiEnv, Player
from reversi_zero.lib import tf_util
from reversi_zero.lib.data_helper import get_game_data_filenames, write_game_data_to_file
from reversi_zero.lib.model_helpler import load_best_model_weight, save_as_best_model, \
    reload_best_model_weight_if_changed, reload_newest_next_generation_model_if_changed

logger = getLogger(__name__)


def start(config: Config):
    tf_util.set_session_config(per_process_gpu_memory_fraction=0.3)
    return SelfPlayWorker(config, env=ReversiEnv()).start()


class SelfPlayWorker:
    def __init__(self, config: Config, env=None, model=None):
        """

        :param config:
        :param ReversiEnv|None env:
        :param reversi_zero.agent.model.ReversiModel|None model:
        """
        self.config = config
        self.model = model
        self.env = env
        self.black = None  # type: ReversiPlayer
        self.white = None  # type: ReversiPlayer
        self.buffer = []
        self.false_positive_count_of_resign = 0
        self.resign_test_game_count = 0

    def start(self):
        if self.model is None:
            self.model = self.load_model()

        self.buffer = []
        idx = 1

        while True:
            start_time = time()
            env = self.start_game(idx)
            end_time = time()
            logger.debug(f"play game {idx} time={end_time - start_time} sec, "
                         f"turn={env.turn}:{env.board.number_of_black_and_white}")
            if True or (idx % self.config.play_data.nb_game_in_file) == 0:
                if self.config.play.use_newest_next_generation_model:
                    reload_newest_next_generation_model_if_changed(self.model)
                else:
                    if reload_best_model_weight_if_changed(self.model):
                        self.reset_false_positive_count()

            idx += 1

    def start_game(self, idx):
        self.env.reset()
        enable_resign = self.config.play.disable_resignation_rate <= random()
        self.black = ReversiPlayer(self.config, self.model, enable_resign=enable_resign)
        self.white = ReversiPlayer(self.config, self.model, enable_resign=enable_resign)
        if not enable_resign:
            logger.debug("Resignation is disabled in the next game.")
        observation = self.env.observation  # type: Board
        while not self.env.done:
            # logger.debug(f"turn={self.env.turn}")
            if self.env.next_player == Player.black:
                action = self.black.action(observation.black, observation.white)
            else:
                action = self.white.action(observation.white, observation.black)
            observation, info = self.env.step(action)
        self.finish_game(resign_enabled=enable_resign)
        self.save_play_data(write=idx % self.config.play_data.nb_game_in_file == 0)
        self.remove_play_data()
        return self.env

    def save_play_data(self, write=True):
        data = self.black.moves + self.white.moves
        self.buffer += data

        if not write:
            return

        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        logger.info(f"save play data to {path}")
        write_game_data_to_file(path, self.buffer)
        self.buffer = []

    def remove_play_data(self):
        files = get_game_data_filenames(self.config.resource)
        if len(files) < self.config.play_data.max_file_num:
            return
        for i in range(len(files) - self.config.play_data.max_file_num):
            os.remove(files[i])

    def finish_game(self, resign_enabled=True):
        if self.env.winner == Winner.black:
            black_win = 1
            false_positive_of_resign = self.black.resigned
        elif self.env.winner == Winner.white:
            black_win = -1
            false_positive_of_resign = self.white.resigned
        else:
            black_win = 0
            false_positive_of_resign = self.black.resigned or self.white.resigned

        self.black.finish_game(black_win)
        self.white.finish_game(-black_win)

        if not resign_enabled:
            self.resign_test_game_count += 1
            if false_positive_of_resign:
                self.false_positive_count_of_resign += 1
                logger.debug("false positive of resignation happened")
            self.check_and_update_resignation_threshold()

    def load_model(self):
        from reversi_zero.agent.model import ReversiModel
        model = ReversiModel(self.config)
        loaded = False
        if not self.config.opts.new:
            if self.config.play.use_newest_next_generation_model:
                loaded = reload_newest_next_generation_model_if_changed(model) or load_best_model_weight(model)
            else:
                loaded = load_best_model_weight(model) or reload_newest_next_generation_model_if_changed(model)

        if not loaded:
            model.build()
            save_as_best_model(model)
        return model

    def reset_false_positive_count(self):
        self.false_positive_count_of_resign = 0
        self.resign_test_game_count = 0

    @property
    def false_positive_rate(self):
        if self.resign_test_game_count == 0:
            return 0
        return self.false_positive_count_of_resign / self.resign_test_game_count

    def check_and_update_resignation_threshold(self):
        if self.resign_test_game_count < 100 or self.config.play.resign_threshold is None:
            return

        old_threshold = self.config.play.resign_threshold
        if self.false_positive_rate >= self.config.play.false_positive_threshold:
            self.config.play.resign_threshold -= self.config.play.resign_threshold_delta
        else:
            self.config.play.resign_threshold += self.config.play.resign_threshold_delta
        logger.debug(f"update resign_threshold: {old_threshold} -> {self.config.play.resign_threshold}")
        self.reset_false_positive_count()

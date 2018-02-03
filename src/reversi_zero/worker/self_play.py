import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from random import random
from time import time
import  numpy as np

from reversi_zero.agent.api import MultiProcessReversiModelAPIServer
from reversi_zero.agent.player import ReversiPlayer
from reversi_zero.config import Config
from reversi_zero.env.reversi_env import Board, Winner
from reversi_zero.env.reversi_env import ReversiEnv, Player
from reversi_zero.lib import tf_util
from reversi_zero.lib.data_helper import get_game_data_filenames, write_game_data_to_file
from reversi_zero.lib.tensorboard_logger import TensorBoardLogger

logger = getLogger(__name__)


def start(config: Config):
    tf_util.set_session_config(per_process_gpu_memory_fraction=0.3)
    api_server = MultiProcessReversiModelAPIServer(config)
    process_num = config.play_data.multi_process_num
    api_server.start_serve()

    with ProcessPoolExecutor(max_workers=process_num) as executor:
        futures = []
        for i in range(process_num):
            play_worker = SelfPlayWorker(config, env=ReversiEnv(), api=api_server.get_api_client(), worker_index=i)
            futures.append(executor.submit(play_worker.start))


class SelfPlayWorker:
    def __init__(self, config: Config, env, api, worker_index=0):
        """

        :param config:
        :param ReversiEnv|None env:
        :param ReversiModelAPI|None api:
        :param int worker_index:
        """
        self.config = config
        self.env = env
        self.api = api
        self.black = None  # type: ReversiPlayer
        self.white = None  # type: ReversiPlayer
        self.buffer = []
        self.false_positive_count_of_resign = 0
        self.resign_test_game_count = 0
        self.worker_index = worker_index
        self.tensor_board = None  # type: TensorBoardLogger

    def start(self):
        logger.debug("SelfPlayWorker#start()")
        np.random.seed(None)
        self.tensor_board = TensorBoardLogger(self.config.resource.self_play_log_dir,
                                              filename_suffix=f"-worker{self.worker_index:03d}")

        self.buffer = []
        idx = self.read_as_int(self.config.resource.self_play_game_idx_file) or 1
        mtcs_info = None

        while True:
            start_time = time()
            if mtcs_info is None and self.config.play.share_mtcs_info_in_self_play:
                mtcs_info = ReversiPlayer.create_mtcs_info()

            # play game
            env = self.start_game(idx, mtcs_info)

            # just log
            end_time = time()
            time_spent = end_time - start_time
            logger.debug(f"play game {idx} time={time_spent} sec, "
                         f"turn={env.turn}:{env.board.number_of_black_and_white}:{env.winner}")

            # log play info to tensor board
            log_info = {"self/time": time_spent, "self/turn": env.turn}
            if mtcs_info:
                log_info["self/mcts_buffer_size"] = len(mtcs_info.var_p)
            self.tensor_board.log_scaler(log_info, idx)

            # reset MCTS info per X games
            if idx % self.config.play.reset_mtcs_info_per_game == 0:
                logger.debug("reset MCTS info")
                mtcs_info = None

            idx += 1
            with open(self.config.resource.self_play_game_idx_file, "wt") as f:
                f.write(str(idx))

    def start_game(self, idx, mtcs_info):
        self.env.reset()
        enable_resign = self.config.play.disable_resignation_rate <= random()
        self.config.play.simulation_num_per_move = self.decide_simulation_num_per_move(idx)
        logger.debug(f"simulation_num_per_move = {self.config.play.simulation_num_per_move}")
        self.black = self.create_reversi_player(enable_resign=enable_resign, mtcs_info=mtcs_info)
        self.white = self.create_reversi_player(enable_resign=enable_resign, mtcs_info=mtcs_info)
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

    def create_reversi_player(self, enable_resign=None, mtcs_info=None):
        return ReversiPlayer(self.config, None, enable_resign=enable_resign, mtcs_info=mtcs_info, api=self.api)

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
        try:
            for i in range(len(files) - self.config.play_data.max_file_num):
                os.remove(files[i])
        except:
            pass

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

    def decide_simulation_num_per_move(self, idx):
        ret = self.read_as_int(self.config.resource.force_simulation_num_file)

        if ret:
            logger.debug(f"loaded simulation num from file: {ret}")
            return ret

        for min_idx, num in self.config.play.schedule_of_simulation_num_per_move:
            if idx >= min_idx:
                ret = num
        return ret

    def read_as_int(self, filename):
        if os.path.exists(filename):
            try:
                with open(filename, "rt") as f:
                    ret = int(str(f.read()).strip())
                    if ret:
                        return ret
            except ValueError:
                pass


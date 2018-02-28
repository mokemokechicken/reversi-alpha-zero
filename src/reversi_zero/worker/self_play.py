import cProfile
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from random import random
from time import time
from traceback import print_stack

import numpy as np
from multiprocessing import Manager, Lock


from reversi_zero.agent.api import MultiProcessReversiModelAPIServer
from reversi_zero.agent.player import ReversiPlayer
from reversi_zero.config import Config
from reversi_zero.env.reversi_env import Board, Winner
from reversi_zero.env.reversi_env import ReversiEnv, Player
from reversi_zero.lib import tf_util
from reversi_zero.lib.data_helper import get_game_data_filenames, write_game_data_to_file
from reversi_zero.lib.file_util import read_as_int
from reversi_zero.lib.ggf import convert_action_to_move, make_ggf_string
from reversi_zero.lib.tensorboard_logger import TensorBoardLogger

logger = getLogger(__name__)


def start(config: Config):
    tf_util.set_session_config(per_process_gpu_memory_fraction=0.3)
    api_server = MultiProcessReversiModelAPIServer(config)
    process_num = config.play_data.multi_process_num
    api_server.start_serve()

    with Manager() as manager:
        shared_var = SharedVar(manager, game_idx=read_as_int(config.resource.self_play_game_idx_file) or 0)
        with ProcessPoolExecutor(max_workers=process_num) as executor:
            futures = []
            for i in range(process_num):
                play_worker = SelfPlayWorker(config, env=ReversiEnv(), api=api_server.get_api_client(),
                                             shared_var=shared_var, worker_index=i)
                futures.append(executor.submit(play_worker.start))


class SharedVar:
    def __init__(self, manager, game_idx: int):
        """

        :param Manager manager:
        :param int game_idx:
        """
        self._lock = manager.Lock()
        self._game_idx = manager.Value('i', game_idx)  # type: multiprocessing.managers.ValueProxy

    @property
    def game_idx(self):
        return self._game_idx.value

    def incr_game_idx(self, n=1):
        with self._lock:
            self._game_idx.value += n
            return self._game_idx.value


class SelfPlayWorker:
    def __init__(self, config: Config, env, api, shared_var, worker_index=0):
        """

        :param config:
        :param ReversiEnv|None env:
        :param ReversiModelAPI|None api:
        :param SharedVar shared_var:
        :param int worker_index:
        """
        self.config = config
        self.env = env
        self.api = api
        self.shared_var = shared_var
        self.black = None  # type: ReversiPlayer
        self.white = None  # type: ReversiPlayer
        self.buffer = []
        self.false_positive_count_of_resign = 0
        self.resign_test_game_count = 0
        self.worker_index = worker_index
        self.tensor_board = None  # type: TensorBoardLogger
        self.move_history = None  # type: MoveHistory
        self.move_history_buffer = []  # type: list[MoveHistory]

    def start(self):
        try:
            self._start()
        except Exception as e:
            print(repr(e))
            print_stack()

    def _start(self):
        logger.debug("SelfPlayWorker#start()")
        np.random.seed(None)
        worker_name = f"worker{self.worker_index:03d}"
        self.tensor_board = TensorBoardLogger(os.path.join(self.config.resource.self_play_log_dir, worker_name))

        self.buffer = []
        mtcs_info = None
        local_idx = 0

        while True:
            np.random.seed(None)
            local_idx += 1
            game_idx = self.shared_var.game_idx

            start_time = time()
            if mtcs_info is None and self.config.play.share_mtcs_info_in_self_play:
                mtcs_info = ReversiPlayer.create_mtcs_info()

            # play game
            env = self.start_game(local_idx, game_idx, mtcs_info)

            game_idx = self.shared_var.incr_game_idx()
            # just log
            end_time = time()
            time_spent = end_time - start_time
            logger.debug(f"play game {game_idx} time={time_spent} sec, "
                         f"turn={env.turn}:{env.board.number_of_black_and_white}:{env.winner}")

            # log play info to tensor board
            prefix = "self"
            log_info = {f"{prefix}/time": time_spent, f"{prefix}/turn": env.turn}
            if mtcs_info:
                log_info[f"{prefix}/mcts_buffer_size"] = len(mtcs_info.var_p)
            self.tensor_board.log_scaler(log_info, game_idx)

            # reset MCTS info per X games
            if self.config.play.reset_mtcs_info_per_game and local_idx % self.config.play.reset_mtcs_info_per_game == 0:
                logger.debug("reset MCTS info")
                mtcs_info = None

            with open(self.config.resource.self_play_game_idx_file, "wt") as f:
                f.write(str(game_idx))

    def start_game(self, local_idx, last_game_idx, mtcs_info):
        # profiler = cProfile.Profile()
        # profiler.enable()

        self.env.reset()
        enable_resign = self.config.play.disable_resignation_rate <= random()
        self.config.play.simulation_num_per_move = self.decide_simulation_num_per_move(last_game_idx)
        logger.debug(f"simulation_num_per_move = {self.config.play.simulation_num_per_move}")
        self.black = self.create_reversi_player(enable_resign=enable_resign, mtcs_info=mtcs_info)
        self.white = self.create_reversi_player(enable_resign=enable_resign, mtcs_info=mtcs_info)
        if not enable_resign:
            logger.debug("Resignation is disabled in the next game.")
        observation = self.env.observation  # type: Board
        self.move_history = MoveHistory()

        # game loop
        while not self.env.done:
            # logger.debug(f"turn={self.env.turn}")
            if self.env.next_player == Player.black:
                action = self.black.action_with_evaluation(observation.black, observation.white)
            else:
                action = self.white.action_with_evaluation(observation.white, observation.black)
            self.move_history.move(self.env, action)
            observation, info = self.env.step(action.action)

        self.finish_game(resign_enabled=enable_resign)
        self.save_play_data(write=local_idx % self.config.play_data.nb_game_in_file == 0)
        self.remove_play_data()

        if self.config.play_data.enable_ggf_data:
            is_write = local_idx % self.config.play_data.nb_game_in_ggf_file == 0
            is_write |= local_idx <= 5
            self.save_ggf_data(write=is_write)

        # profiler.disable()
        # profiler.dump_stats(f"profile-worker-{self.worker_index}-{local_idx}")
        return self.env

    def create_reversi_player(self, enable_resign=None, mtcs_info=None):
        return ReversiPlayer(self.config, None, enable_resign=enable_resign, mtcs_info=mtcs_info, api=self.api)

    def save_play_data(self, write=True):
        # drop draw game by drop_draw_game_rate
        if self.black.moves[0][-1] != 0 or self.config.play_data.drop_draw_game_rate <= np.random.random():
            data = self.black.moves + self.white.moves
            self.buffer += data

        if not write or not self.buffer:
            return

        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        logger.info(f"save play data to {path}")
        write_game_data_to_file(path, self.buffer)
        self.buffer = []

    def save_ggf_data(self, write=True):
        self.move_history_buffer.append(self.move_history)
        if not write:
            return

        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.self_play_ggf_data_dir, rc.ggf_filename_tmpl % game_id)
        with open(path, "wt") as f:
            for mh in self.move_history_buffer:
                f.write(mh.make_ggf_string("RAZ", "RAZ") + "\n")
        self.move_history_buffer = []

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
        ret = read_as_int(self.config.resource.force_simulation_num_file)

        if ret:
            logger.debug(f"loaded simulation num from file: {ret}")
            return ret

        for min_idx, num in self.config.play.schedule_of_simulation_num_per_move:
            if idx >= min_idx:
                ret = num
        return ret


class MoveHistory:
    def __init__(self):
        self.moves = []

    def move(self, env, action):
        """

        :param ReversiEnv env:
        :param ActionWithEvaluation action:
        :return:
        """
        if action.action is None:
            return  # resigned

        if len(self.moves) % 2 == 0:
            if env.next_player == Player.white:
                self.moves.append(convert_action_to_move(None))
        else:
            if env.next_player == Player.black:
                self.moves.append(convert_action_to_move(None))
        move = f"{convert_action_to_move(action.action)}/{action.q*10}/{action.n}"
        self.moves.append(move)

    def make_ggf_string(self, black_name=None, white_name=None):
        return make_ggf_string(black_name=black_name, white_name=white_name, moves=self.moves)

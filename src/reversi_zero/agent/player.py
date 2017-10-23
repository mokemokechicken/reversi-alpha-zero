from collections import defaultdict, namedtuple
from logging import getLogger

import numpy as np
from numpy.random import random

from reversi_zero.agent.api import ReversiModelAPI
from reversi_zero.config import Config
from reversi_zero.env.reversi_env import ReversiEnv, Player
from reversi_zero.lib.bitboard import find_correct_moves, bit_to_array, flip_vertical, rotate90

CounterKey = namedtuple("CounterKey", "black white next_player")

logger = getLogger(__name__)


class ReversiPlayer:
    def __init__(self, config: Config, model, play_config=None):
        """

        :param config:
        :param reversi_zero.agent.model.ReversiModel model:
        """
        self.config = config
        self.model = model
        self.play_config = play_config or self.config.play
        self.api = ReversiModelAPI(self.config, self.model)

        # key=(own, enemy, action)
        self.var_n = defaultdict(lambda: np.zeros((64,)))
        self.var_w = defaultdict(lambda: np.zeros((64,)))
        self.var_q = defaultdict(lambda: np.zeros((64,)))
        self.var_u = defaultdict(lambda: np.zeros((64,)))
        self.var_p = defaultdict(lambda: np.zeros((64,)))
        self.expanded = set()
        self.moves = []

    def action(self, own, enemy):
        """

        :param own: BitBoard
        :param enemy:  BitBoard
        :return: action: move pos=0 ~ 63 (0=top left, 7 top right, 63 bottom right)
        """
        for it in range(self.play_config.simulation_num_per_move):
            self.search_my_move(ReversiEnv().update(own, enemy, Player.black), is_root_node=True)
        policy = self.calc_policy(own, enemy)
        self.moves.append([(own, enemy), list(policy)])
        return int(np.random.choice(range(64), p=policy))

    def search_my_move(self, env: ReversiEnv, is_root_node=False):
        """

        Q, V is value for this Player(always black).
        P is value for the player of next_player (black or white)
        :param env:
        :param is_root_node:
        :return:
        """
        if env.done:
            if env.winner == Player.black:
                return 1
            elif env.winner == Player.white:
                return -1
            else:
                return 0

        key = self.counter_key(env)

        # is leaf?
        if key not in self.expanded:  # reach leaf node
            leaf_v = self.expand_and_evaluate(env)
            if env.next_player == Player.black:
                return leaf_v  # Value for black
            else:
                return -leaf_v  # Value for white == -Value for black
        else:
            action_t = self.select_action_q_and_u(env, is_root_node)
            _, _ = env.step(action_t)
            leaf_v = self.search_my_move(env)  # next move

        # on returning search path
        # update: N, W, Q, U
        n = self.var_n[key][action_t] = self.var_n[key][action_t] + 1
        w = self.var_w[key][action_t] = self.var_w[key][action_t] + leaf_v
        self.var_q[key][action_t] = w / n
        return leaf_v

    def finish_game(self, z):
        """

        :param z: win=1, lose=-1, draw=0
        :return:
        """
        for move in self.moves:
            move += [z]

    def calc_policy(self, own, enemy):
        """calc π(a|s0)

        :param own:
        :param enemy:
        :return:
        """
        pc = self.play_config
        env = ReversiEnv().update(own, enemy, Player.black)
        key = self.counter_key(env)
        if env.turn < pc.change_tau_turn:
            return self.var_n[key] / np.sum(self.var_n[key])  # tau = 1
        else:
            action = np.argmax(self.var_n[key])  # tau = 0
            ret = np.zeros(64)
            ret[action] = 1
            return ret

    @staticmethod
    def counter_key(env: ReversiEnv):
        return CounterKey(env.board.black, env.board.white, env.next_player.value)

    def select_action_q_and_u(self, env, is_root_node):
        key = self.counter_key(env)
        if env.next_player == Player.black:
            legal_moves = find_correct_moves(key.black, key.white)
        else:
            legal_moves = find_correct_moves(key.white, key.black)
        # noinspection PyUnresolvedReferences
        xx_ = np.sqrt(np.sum(self.var_n[key]))  # SQRT of sum(N(s, b); for all b)
        xx_ = max(xx_, 1)  # avoid u_=0 if N is all 0
        p_ = self.var_p[key]

        if is_root_node:  # Is it correct?? -> (1-e)p + e*Dir(0.03)
            p_ = (1 - self.play_config.noise_eps) * p_ + \
                 self.play_config.noise_eps * np.random.dirichlet([self.play_config.dirichlet_alpha] * 64)

        u_ = self.play_config.c_puct * p_ * xx_ / (1 + self.var_n[key])
        if env.next_player == Player.black:
            v_ = (self.var_q[key] + u_ + 1000) * bit_to_array(legal_moves, 64)
        else:
            v_ = (-self.var_q[key] + u_ + 1000) * bit_to_array(legal_moves, 64)

        # noinspection PyTypeChecker
        action_t = int(np.argmax(v_))
        return action_t

    def expand_and_evaluate(self, env):
        """新しいleaf, doneの場合もある

        update var_p, return leaf_v

        :param ReversiEnv env:
        :return: leaf_v
        """

        key = self.counter_key(env)
        self.expanded.add(key)

        black, white = env.board.black, env.board.white
        if random() < 0.5:
            black, white = flip_vertical(black), flip_vertical(white)
        for i in range(int(random() * 4)):
            black, white = rotate90(black), rotate90(white)

        black_ary = bit_to_array(black, 64).reshape((8, 8))
        white_ary = bit_to_array(white, 64).reshape((8, 8))

        if env.next_player == Player.black:
            leaf_p, leaf_v = self.api.predict(np.array([black_ary, white_ary]))
        else:
            leaf_p, leaf_v = self.api.predict(np.array([white_ary, black_ary]))

        self.var_p[key] = leaf_p  # P is value for next_player (black or white)
        return float(leaf_v)

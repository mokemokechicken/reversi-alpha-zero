from _asyncio import Future
from asyncio.queues import Queue
from collections import defaultdict, namedtuple
from logging import getLogger
import asyncio

import numpy as np
from numpy.random import random

from reversi_zero.agent.api import ReversiModelAPI
from reversi_zero.config import Config
from reversi_zero.env.reversi_env import ReversiEnv, Player
from reversi_zero.lib.bitboard import find_correct_moves, bit_to_array, flip_vertical, rotate90

CounterKey = namedtuple("CounterKey", "black white next_player")
QueueItem = namedtuple("QueueItem", "state future")
HistoryItem = namedtuple("HistoryItem", "action policy value")

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
        self.now_expanding = set()
        self.prediction_queue = Queue(self.play_config.prediction_queue_size)
        self.sem = asyncio.Semaphore(self.play_config.parallel_search_num)

        self.moves = []
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0

        self.thinking_history = {}  # for fun

    def action(self, own, enemy):
        """

        :param own: BitBoard
        :param enemy:  BitBoard
        :return: action: move pos=0 ~ 63 (0=top left, 7 top right, 63 bottom right)
        """
        estimated_value = self.search_moves(own, enemy)
        policy = self.calc_policy(own, enemy)
        self.moves.append([(own, enemy), list(policy)])
        action = int(np.random.choice(range(64), p=policy))
        self.thinking_history[(own, enemy)] = HistoryItem(action, policy, estimated_value)
        return action

    def ask_thought_about(self, own, enemy) -> HistoryItem:
        return self.thinking_history.get((own, enemy))

    def search_moves(self, own, enemy):
        loop = self.loop
        self.running_simulation_num = 0

        coroutine_list = []
        for it in range(self.play_config.simulation_num_per_move):
            cor = self.start_search_my_move(own, enemy)
            coroutine_list.append(cor)

        coroutine_list.append(self.prediction_worker())
        leaf_v_list = loop.run_until_complete(asyncio.gather(*coroutine_list))
        return float(np.average([v for v in leaf_v_list if v is not None]))

    async def start_search_my_move(self, own, enemy):
        self.running_simulation_num += 1
        with await self.sem:  # reduce parallel search number
            env = ReversiEnv().update(own, enemy, Player.black)
            leaf_v = await self.search_my_move(env, is_root_node=True)
            self.running_simulation_num -= 1
            return leaf_v

    async def search_my_move(self, env: ReversiEnv, is_root_node=False):
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

        while key in self.now_expanding:
            await asyncio.sleep(self.config.play.wait_for_expanding_sleep_sec)

        # is leaf?
        if key not in self.expanded:  # reach leaf node
            leaf_v = await self.expand_and_evaluate(env)
            if env.next_player == Player.black:
                return leaf_v  # Value for black
            else:
                return -leaf_v  # Value for white == -Value for black

        action_t = self.select_action_q_and_u(env, is_root_node)
        _, _ = env.step(action_t)

        virtual_loss = self.config.play.virtual_loss
        self.var_n[key][action_t] += virtual_loss
        self.var_w[key][action_t] -= virtual_loss
        leaf_v = await self.search_my_move(env)  # next move

        # on returning search path
        # update: N, W, Q, U
        n = self.var_n[key][action_t] = self.var_n[key][action_t] - virtual_loss + 1
        w = self.var_w[key][action_t] = self.var_w[key][action_t] + virtual_loss + leaf_v
        self.var_q[key][action_t] = w / n
        return leaf_v

    async def expand_and_evaluate(self, env):
        """新しいleaf, doneの場合もある

        update var_p, return leaf_v

        :param ReversiEnv env:
        :return: leaf_v
        """

        key = self.counter_key(env)
        self.now_expanding.add(key)

        black, white = env.board.black, env.board.white
        if random() < 0.5:
            black, white = flip_vertical(black), flip_vertical(white)
        for i in range(int(random() * 4)):
            black, white = rotate90(black), rotate90(white)

        black_ary = bit_to_array(black, 64).reshape((8, 8))
        white_ary = bit_to_array(white, 64).reshape((8, 8))
        state = [black_ary, white_ary] if env.next_player == Player.black else [white_ary, black_ary]
        future = await self.predict(np.array(state))  # type: Future
        await future
        leaf_p, leaf_v = future.result()

        self.var_p[key] = leaf_p  # P is value for next_player (black or white)
        self.expanded.add(key)
        self.now_expanding.remove(key)
        return float(leaf_v)

    async def prediction_worker(self):
        q = self.prediction_queue
        margin = 10
        while self.running_simulation_num > 0 or margin > 0:
            if q.empty():
                if margin > 0:
                    margin -= 1
                await asyncio.sleep(self.config.play.prediction_worker_sleep_sec)
                continue
            item_list = [q.get_nowait() for _ in range(q.qsize())]  # type: list[QueueItem]
            # logger.debug(f"predicting {len(item_list)} items")
            data = np.array([x.state for x in item_list])
            policy_ary, value_ary = self.api.predict(data)
            for p, v, item in zip(policy_ary, value_ary, item_list):
                item.future.set_result((p, v))

    async def predict(self, x):
        future = self.loop.create_future()
        item = QueueItem(x, future)
        await self.prediction_queue.put(item)
        return future

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

from _asyncio import Future
from asyncio.queues import Queue
from collections import defaultdict, namedtuple
from logging import getLogger
import asyncio

import numpy as np
from numpy.random import random

from reversi_zero.agent.api import ReversiModelAPI
from reversi_zero.config import Config
from reversi_zero.env.reversi_env import ReversiEnv, Player, Winner
from reversi_zero.lib.bitboard import find_correct_moves, bit_to_array, flip_vertical, rotate90, dirichlet_noise_of_mask

CounterKey = namedtuple("CounterKey", "black white next_player")
QueueItem = namedtuple("QueueItem", "state future")
HistoryItem = namedtuple("HistoryItem", "action policy values visit enemy_values enemy_visit")
CallbackInMCTS = namedtuple("CallbackInMCTS", "per_sim callback")
MCTSInfo = namedtuple("MCTSInfo", "var_n var_w var_p")

logger = getLogger(__name__)


class ReversiPlayer:
    def __init__(self, config: Config, model, play_config=None, enable_resign=True, mtcs_info=None):
        """

        :param config:
        :param reversi_zero.agent.model.ReversiModel model:
        :param MCTSInfo mtcs_info:
        """
        self.config = config
        self.model = model
        self.play_config = play_config or self.config.play
        self.enable_resign = enable_resign
        self.api = ReversiModelAPI(self.config, self.model)

        # key=(own, enemy, action)
        mtcs_info = mtcs_info or self.create_mtcs_info()
        self.var_n, self.var_w, self.var_p = mtcs_info

        self.expanded = set()
        self.now_expanding = set()
        self.prediction_queue = Queue(self.play_config.prediction_queue_size)
        self.sem = asyncio.Semaphore(self.play_config.parallel_search_num)

        self.moves = []
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0
        self.callback_in_mtcs = None

        self.thinking_history = {}  # for fun
        self.resigned = False
        self.requested_stop_thinking = False

    @staticmethod
    def create_mtcs_info():
        return MCTSInfo(defaultdict(lambda: np.zeros((64,))),
                        defaultdict(lambda: np.zeros((64,))),
                        defaultdict(lambda: np.zeros((64,))))

    def var_q(self, key):
        return self.var_w[key] / (self.var_n[key] + 1e-5)

    def action(self, own, enemy, callback_in_mtcs=None):
        """

        :param own: BitBoard
        :param enemy:  BitBoard
        :param CallbackInMCTS callback_in_mtcs:
        :return: action: move pos=0 ~ 63 (0=top left, 7 top right, 63 bottom right)
        """
        env = ReversiEnv().update(own, enemy, Player.black)
        key = self.counter_key(env)
        self.callback_in_mtcs = callback_in_mtcs

        for tl in range(self.play_config.thinking_loop):
            if tl > 0 and self.play_config.logging_thinking:
                logger.debug(f"continue thinking: policy move=({action % 8}, {action // 8}), "
                             f"value move=({action_by_value % 8}, {action_by_value // 8})")
            self.search_moves(own, enemy)
            policy = self.calc_policy(own, enemy)
            action = int(np.random.choice(range(64), p=policy))
            action_by_value = int(np.argmax(self.var_q(key) + (self.var_n[key] > 0)*100))
            if action == action_by_value or env.turn < self.play_config.change_tau_turn or env.turn <= 1:
                break

        # this is for play_gui, not necessary when training.
        next_key = self.get_next_key(own, enemy, action)
        self.thinking_history[(own, enemy)] = HistoryItem(action, policy, list(self.var_q(key)), list(self.var_n[key]),
                                                          list(self.var_q(next_key)), list(self.var_n[next_key]))

        if self.play_config.resign_threshold is not None and\
                        np.max(self.var_q(key) - (self.var_n[key] == 0)*10) <= self.play_config.resign_threshold:
            self.resigned = True
            if self.enable_resign:
                if env.turn >= self.config.play.allowed_resign_turn:
                    return None  # means resign
                else:
                    logger.debug(f"Want to resign but disallowed turn {env.turn} < {self.config.play.allowed_resign_turn}")

        saved_policy = self.calc_policy_by_tau_1(key) if self.config.play_data.save_policy_of_tau_1 else policy
        self.add_data_to_move_buffer_with_8_symmetries(own, enemy, saved_policy)
        return action

    def stop_thinking(self):
        self.requested_stop_thinking = True

    def add_data_to_move_buffer_with_8_symmetries(self, own, enemy, policy):
        for flip in [False, True]:
            for rot_right in range(4):
                own_saved, enemy_saved, policy_saved = own, enemy, policy.reshape((8, 8))
                if flip:
                    own_saved = flip_vertical(own_saved)
                    enemy_saved = flip_vertical(enemy_saved)
                    policy_saved = np.flipud(policy_saved)
                if rot_right:
                    for _ in range(rot_right):
                        own_saved = rotate90(own_saved)
                        enemy_saved = rotate90(enemy_saved)
                    policy_saved = np.rot90(policy_saved, k=-rot_right)
                self.moves.append([(own_saved, enemy_saved), list(policy_saved.reshape((64, )))])

    def get_next_key(self, own, enemy, action):
        env = ReversiEnv().update(own, enemy, Player.black)
        env.step(action)
        return self.counter_key(env)

    def ask_thought_about(self, own, enemy) -> HistoryItem:
        return self.thinking_history.get((own, enemy))

    def search_moves(self, own, enemy):
        loop = self.loop
        self.running_simulation_num = 0
        self.requested_stop_thinking = False

        coroutine_list = []
        for it in range(self.play_config.simulation_num_per_move):
            cor = self.start_search_my_move(own, enemy)
            coroutine_list.append(cor)

        coroutine_list.append(self.prediction_worker())
        loop.run_until_complete(asyncio.gather(*coroutine_list))

    async def start_search_my_move(self, own, enemy):
        self.running_simulation_num += 1
        root_key = self.counter_key(ReversiEnv().update(own, enemy, Player.black))
        with await self.sem:  # reduce parallel search number
            if self.requested_stop_thinking:
                self.running_simulation_num -= 1
                return None
            env = ReversiEnv().update(own, enemy, Player.black)
            leaf_v = await self.search_my_move(env, is_root_node=True)
            self.running_simulation_num -= 1
            if self.callback_in_mtcs and self.callback_in_mtcs.per_sim > 0 and \
                    self.running_simulation_num % self.callback_in_mtcs.per_sim == 0:
                self.callback_in_mtcs.callback(list(self.var_q(root_key)), list(self.var_n[root_key]))
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
            if env.winner == Winner.black:
                return 1
            elif env.winner == Winner.white:
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

        virtual_loss = self.config.play.virtual_loss
        virtual_loss_for_w = virtual_loss if env.next_player == Player.black else -virtual_loss

        action_t = self.select_action_q_and_u(env, is_root_node)
        _, _ = env.step(action_t)

        self.var_n[key][action_t] += virtual_loss
        self.var_w[key][action_t] -= virtual_loss_for_w
        leaf_v = await self.search_my_move(env)  # next move

        # on returning search path
        # update: N, W
        self.var_n[key][action_t] += - virtual_loss + 1
        self.var_w[key][action_t] += virtual_loss_for_w + leaf_v
        return leaf_v

    async def expand_and_evaluate(self, env):
        """expand new leaf

        update var_p, return leaf_v

        :param ReversiEnv env:
        :return: leaf_v
        """

        key = self.counter_key(env)
        self.now_expanding.add(key)

        black, white = env.board.black, env.board.white

        # (di(p), v) = fθ(di(sL))
        # rotation and flip. flip -> rot.
        is_flip_vertical = random() < 0.5
        rotate_right_num = int(random() * 4)
        if is_flip_vertical:
            black, white = flip_vertical(black), flip_vertical(white)
        for i in range(rotate_right_num):
            black, white = rotate90(black), rotate90(white)  # rotate90: rotate bitboard RIGHT 1 time

        black_ary = bit_to_array(black, 64).reshape((8, 8))
        white_ary = bit_to_array(white, 64).reshape((8, 8))
        state = [black_ary, white_ary] if env.next_player == Player.black else [white_ary, black_ary]
        future = await self.predict(np.array(state))  # type: Future
        await future
        leaf_p, leaf_v = future.result()

        # reverse rotate and flip about leaf_p
        if rotate_right_num > 0 or is_flip_vertical:  # reverse rotation and flip. rot -> flip.
            leaf_p = leaf_p.reshape((8, 8))
            if rotate_right_num > 0:
                leaf_p = np.rot90(leaf_p, k=rotate_right_num)  # rot90: rotate matrix LEFT k times
            if is_flip_vertical:
                leaf_p = np.flipud(leaf_p)
            leaf_p = leaf_p.reshape((64, ))

        self.var_p[key] = leaf_p  # P is value for next_player (black or white)
        self.expanded.add(key)
        self.now_expanding.remove(key)
        return float(leaf_v)

    async def prediction_worker(self):
        """For better performance, queueing prediction requests and predict together in this worker.

        speed up about 45sec -> 15sec for example.
        :return:
        """
        q = self.prediction_queue
        margin = 10  # avoid finishing before other searches starting.
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
        for move in self.moves:  # add this game winner result to all past moves.
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
            return self.calc_policy_by_tau_1(key)
        else:
            action = np.argmax(self.var_n[key])  # tau = 0
            ret = np.zeros(64)
            ret[action] = 1
            return ret

    def calc_policy_by_tau_1(self, key):
        return self.var_n[key] / np.sum(self.var_n[key])  # tau = 1

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

        if is_root_node and self.play_config.noise_eps > 0:  # Is it correct?? -> (1-e)p + e*Dir(alpha)
            if self.play_config.dirichlet_noise_only_for_legal_moves:
                noise = dirichlet_noise_of_mask(legal_moves, self.play_config.dirichlet_alpha)
            else:
                noise = np.random.dirichlet([self.play_config.dirichlet_alpha] * 64)
            p_ = (1 - self.play_config.noise_eps) * p_ + self.play_config.noise_eps * noise

        # re-normalize in legal moves
        p_ = p_ * bit_to_array(legal_moves, 64)
        if np.sum(p_) > 0:
            p_ = p_ / np.sum(p_)

        u_ = self.play_config.c_puct * p_ * xx_ / (1 + self.var_n[key])
        if env.next_player == Player.black:
            v_ = (self.var_q(key) + u_ + 1000) * bit_to_array(legal_moves, 64)
        else:
            # When enemy's selecting action, flip Q-Value.
            v_ = (-self.var_q(key) + u_ + 1000) * bit_to_array(legal_moves, 64)

        # noinspection PyTypeChecker
        action_t = int(np.argmax(v_))
        return action_t

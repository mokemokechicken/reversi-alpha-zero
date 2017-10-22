import os
from datetime import datetime
from logging import getLogger

import keras.backend as K
import numpy as np
from keras.optimizers import SGD

from reversi_zero.agent.model import ReversiModel, objective_function_for_policy, \
    objective_function_for_value
from reversi_zero.config import Config
from reversi_zero.lib.bitboard import bit_to_array
from reversi_zero.lib.data_helper import get_game_data_filenames, read_game_data_from_file
from reversi_zero.lib.model_helpler import load_best_model_weight

logger = getLogger(__name__)


def start(config: Config):
    return OptimizeWorker(config).start()


class OptimizeWorker:
    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: ReversiModel
        self.loaded_filenames = set()
        self.loaded_data = {}
        self.dataset = None
        self.optimizer = None

    def start(self):
        self.model = self.load_model()
        self.training()

    def training(self):
        self.compile_model()
        total_steps = self.config.trainer.start_total_steps

        while True:
            self.load_play_data()
            self.update_learning_rate(total_steps)  # TODO: 中断からの再開をどう扱うか
            steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
            total_steps += steps

            self.save_current_model()

    def train_epoch(self, epochs):
        tc = self.config.trainer
        state_ary, policy_ary, z_ary = self.dataset
        self.model.model.fit(state_ary, [policy_ary, z_ary],
                             batch_size=tc.batch_size,
                             epochs=epochs)
        steps = (state_ary.shape[0] // tc.batch_size) * epochs
        return steps

    def compile_model(self):
        self.optimizer = SGD(lr=1e-2, momentum=0.9)
        losses = [objective_function_for_policy, objective_function_for_value]
        self.model.model.compile(optimizer=self.optimizer, loss=losses)

    def update_learning_rate(self, total_steps):
        # The deepmind paper says
        # ~400k: 1e-2
        # 400k~600k: 1e-3
        # 600k~: 1e-4
        if total_steps < 400000:
            K.set_value(self.optimizer.lr, 1e-2)
        elif total_steps < 600000:
            K.set_value(self.optimizer.lr, 1e-3)
        else:
            K.set_value(self.optimizer.lr, 1e-4)

    def save_current_model(self):
        rc = self.config.resource
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        config_path = os.path.join(rc.next_generation_model_dir,
                                   rc.next_generation_model_config_filename_tmpl % model_id)
        weight_path = os.path.join(rc.next_generation_model_dir,
                                   rc.next_generation_model_weight_filename_tmpl % model_id)
        self.model.save(config_path, weight_path)

    def collect_all_loaded_data(self):
        state_ary_list, policy_ary_list, z_ary_list = [], [], []
        for s_ary, p_ary, z_ary_ in self.loaded_data.values():
            state_ary_list.append(s_ary)
            policy_ary_list.append(p_ary)
            z_ary_list.append(z_ary_)

        state_ary = np.concatenate(state_ary_list)
        policy_ary = np.concatenate(policy_ary_list)
        z_ary = np.concatenate(z_ary_list)
        return state_ary, policy_ary, z_ary

    def load_model(self):
        from reversi_zero.agent.model import ReversiModel
        model = ReversiModel(self.config)
        if not load_best_model_weight(model):
            raise RuntimeError(f"Best model can not loaded!")
        return model

    def load_play_data(self):
        filenames = get_game_data_filenames(self.config.resource)
        updated = False
        for filename in filenames:
            if filename in self.loaded_filenames:
                continue
            self.load_data_from_file(filename)
            updated = True

        for filename in (self.loaded_filenames - set(filenames)):
            self.unload_data_of_file(filename)
            updated = True

        if updated:
            self.dataset = self.collect_all_loaded_data()

    def load_data_from_file(self, filename):
        try:
            logger.debug(f"loading data from {filename}")
            data = read_game_data_from_file(filename)
            self.loaded_data[filename] = self.convert_to_training_data(data)
        except Exception as e:
            logger.warning(str(e))

    def unload_data_of_file(self, filename):
        logger.debug(f"removing data about {filename} from training set")
        self.loaded_filenames.remove(filename)
        if filename in self.loaded_data:
            del self.loaded_data[filename]

    @staticmethod
    def convert_to_training_data(data):
        """

        :param data: format is SelfPlayWorker.buffer
            list of [(own: bitboard, enemy: bitboard), [policy: float 64 items], z: number]
        :return:
        """
        state_list = []
        policy_list = []
        z_list = []
        for state, policy, z in data:
            own, enemy = bit_to_array(state[0], 64).reshape((8, 8)), bit_to_array(state[1], 64).reshape((8, 8))
            state_list.append([own, enemy])
            policy_list.append(policy)
            z_list.append(z)

        return np.array(state_list), np.array(policy_list), np.array(z_list)

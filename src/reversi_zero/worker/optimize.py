import os
from collections import Counter
from datetime import datetime
from logging import getLogger
from time import sleep, time

import keras.backend as K
import numpy as np
from keras.callbacks import Callback
from keras.optimizers import SGD

from reversi_zero.agent.model import ReversiModel, objective_function_for_policy, \
    objective_function_for_value
from reversi_zero.config import Config
from reversi_zero.lib import tf_util
from reversi_zero.lib.bitboard import bit_to_array
from reversi_zero.lib.data_helper import get_game_data_filenames, read_game_data_from_file, \
    get_next_generation_model_dirs
from reversi_zero.lib.model_helpler import load_best_model_weight
from reversi_zero.lib.tensorboard_step_callback import TensorBoardStepCallback

logger = getLogger(__name__)


def start(config: Config):
    tf_util.set_session_config(per_process_gpu_memory_fraction=0.65)
    return OptimizeWorker(config).start()


class OptimizeWorker:
    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: ReversiModel
        self.loaded_filenames = set()
        self.loaded_data = {}
        self.training_count_of_files = Counter()
        self.dataset = None
        self.optimizer = None

    def start(self):
        self.model = self.load_model()
        self.training()

    def training(self):
        self.compile_model()
        total_steps = self.config.trainer.start_total_steps
        save_model_callback = PerStepCallback(self.config.trainer.save_model_steps, self.save_current_model,
                                              self.config.trainer.wait_after_save_model_ratio)
        callbacks = [save_model_callback]  # type: list[Callback]
        tb_callback = None  # type: TensorBoardStepCallback

        if self.config.trainer.use_tensorboard:
            tb_callback = TensorBoardStepCallback(
                log_dir=self.config.resource.tensorboard_log_dir,
                logging_per_steps=self.config.trainer.logging_per_steps,
                step=total_steps,
            )
            callbacks.append(tb_callback)

        while True:
            self.load_play_data()
            if self.dataset_size < self.config.trainer.min_data_size_to_learn:
                logger.info(f"dataset_size={self.dataset_size} is less than {self.config.trainer.min_data_size_to_learn}")
                sleep(10)
                continue
            self.update_learning_rate(total_steps)
            total_steps += self.train_epoch(self.config.trainer.epoch_to_checkpoint, callbacks)
            self.count_up_training_count_and_delete_self_play_data_files()

        if tb_callback:  # This code is never reached. But potentially this is required.
            tb_callback.close()

    def train_epoch(self, epochs, callbacks):
        tc = self.config.trainer
        state_ary, policy_ary, z_ary = self.dataset
        self.model.model.fit(state_ary, [policy_ary, z_ary],
                             batch_size=tc.batch_size,
                             callbacks=callbacks,
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

        lr = self.decide_learning_rate(total_steps)
        if lr:
            K.set_value(self.optimizer.lr, lr)
            logger.debug(f"total step={total_steps}, set learning rate to {lr}")

    def decide_learning_rate(self, total_steps):
        ret = None

        if os.path.exists(self.config.resource.force_learing_rate_file):
            try:
                with open(self.config.resource.force_learing_rate_file, "rt") as f:
                    ret = float(str(f.read()).strip())
                    if ret:
                        logger.debug(f"loaded lr from force learning rate file: {ret}")
                        return ret
            except ValueError:
                pass

        for step, lr in self.config.trainer.lr_schedules:
            if total_steps >= step:
                ret = lr
        return ret

    def save_current_model(self):
        rc = self.config.resource
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        model_dir = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % model_id)
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        self.model.save(config_path, weight_path)

    def collect_all_loaded_data(self):
        state_ary_list, policy_ary_list, z_ary_list = [], [], []
        for s_ary, p_ary, z_ary_ in self.loaded_data.values():
            state_ary_list.append(s_ary)
            policy_ary_list.append(p_ary)
            z_ary_list.append(z_ary_)

        if state_ary_list:
            state_ary = np.concatenate(state_ary_list)
            policy_ary = np.concatenate(policy_ary_list)
            z_ary = np.concatenate(z_ary_list)
            return state_ary, policy_ary, z_ary
        else:
            return None

    @property
    def dataset_size(self):
        if self.dataset is None:
            return 0
        return len(self.dataset[0])

    def load_model(self):
        from reversi_zero.agent.model import ReversiModel
        model = ReversiModel(self.config)
        rc = self.config.resource

        dirs = get_next_generation_model_dirs(rc)
        if not dirs:
            logger.debug(f"loading best model")
            if not load_best_model_weight(model):
                raise RuntimeError(f"Best model can not loaded!")
        else:
            latest_dir = dirs[-1]
            logger.debug(f"loading latest model")
            config_path = os.path.join(latest_dir, rc.next_generation_model_config_filename)
            weight_path = os.path.join(latest_dir, rc.next_generation_model_weight_filename)
            model.load(config_path, weight_path)
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
            logger.debug("updating training dataset")
            self.dataset = self.collect_all_loaded_data()

    def load_data_from_file(self, filename):
        try:
            logger.debug(f"loading data from {filename}")
            data = read_game_data_from_file(filename)
            self.loaded_data[filename] = self.convert_to_training_data(data)
            self.loaded_filenames.add(filename)
        except Exception as e:
            logger.warning(str(e))

    def unload_data_of_file(self, filename):
        logger.debug(f"removing data about {filename} from training set")
        self.loaded_filenames.remove(filename)
        if filename in self.loaded_data:
            del self.loaded_data[filename]
        if filename in self.training_count_of_files:
            del self.training_count_of_files[filename]

    def count_up_training_count_and_delete_self_play_data_files(self):
        limit = self.config.trainer.delete_self_play_after_number_of_training
        if not limit:
            return

        for filename in self.loaded_filenames:
            self.training_count_of_files[filename] += 1
            if self.training_count_of_files[filename] >= limit:
                if os.path.exists(filename):
                    try:
                        logger.debug(f"remove {filename}")
                        os.remove(filename)
                    except Exception as e:
                        logger.warning(e)

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


class PerStepCallback(Callback):
    def __init__(self, per_step, callback, wait_after_save_model_ratio=None):
        super().__init__()
        self.per_step = per_step
        self.step = 0
        self.callback = callback
        self.wait_after_save_model_ratio = wait_after_save_model_ratio
        self.last_wait_time = time()

    def on_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step % self.per_step == 0:
            self.callback()
            self.wait()

    def wait(self):
        if self.wait_after_save_model_ratio:
            time_spent = time() - self.last_wait_time
            logger.debug(f"start sleeping {time_spent} seconds")
            sleep(time_spent * self.wait_after_save_model_ratio)
            logger.debug(f"finish sleeping")
            self.last_wait_time = time()

import numpy as np

from multiprocessing import Pipe, connection
from threading import Thread
from time import time

from logging import getLogger

from reversi_zero.agent.model import ReversiModel
from reversi_zero.config import Config

from reversi_zero.lib.model_helpler import reload_newest_next_generation_model_if_changed, load_best_model_weight, \
    save_as_best_model, reload_best_model_weight_if_changed
import tensorflow as tf


logger = getLogger(__name__)


class ReversiModelAPI:
    def __init__(self, config: Config, agent_model):
        """

        :param config:
        :param reversi_zero.agent.model.ReversiModel agent_model:
        """
        self.config = config
        self.agent_model = agent_model

    def predict(self, x):
        assert x.ndim in (3, 4)
        assert x.shape == (2, 8, 8) or x.shape[1:] == (2, 8, 8)
        orig_x = x
        if x.ndim == 3:
            x = x.reshape(1, 2, 8, 8)

        policy, value = self._do_predict(x)

        if orig_x.ndim == 3:
            return policy[0], value[0]
        else:
            return policy, value

    def _do_predict(self, x):
        return self.agent_model.model.predict_on_batch(x)


class MultiProcessReversiModelAPIServer:
    # https://github.com/Akababa/Chess-Zero/blob/nohistory/src/chess_zero/agent/api_chess.py

    def __init__(self, config: Config):
        """

        :param config:
        """
        self.config = config
        self.model = None  # type: ReversiModel
        self.connections = []

    def get_api_client(self):
        me, you = Pipe()
        self.connections.append(me)
        return MultiProcessReversiModelAPIClient(self.config, None, you)

    def start_serve(self):
        self.model = self.load_model()
        # threading workaround: https://github.com/keras-team/keras/issues/5640
        self.model.model._make_predict_function()
        self.graph = tf.get_default_graph()

        prediction_worker = Thread(target=self.prediction_worker, name="prediction_worker")
        prediction_worker.daemon = True
        prediction_worker.start()

    def prediction_worker(self):
        logger.debug("prediction_worker started")
        average_prediction_size = []
        last_model_check_time = time()
        while True:
            if last_model_check_time+60 < time():
                self.try_reload_model()
                last_model_check_time = time()
                logger.debug(f"average_prediction_size={np.average(average_prediction_size)}")
                average_prediction_size = []
            ready_conns = connection.wait(self.connections, timeout=0.001)  # type: list[Connection]
            if not ready_conns:
                continue
            data = []
            size_list = []
            for conn in ready_conns:
                x = conn.recv()
                data.append(x)  # shape: (k, 2, 8, 8)
                size_list.append(x.shape[0])  # save k
            average_prediction_size.append(np.sum(size_list))
            array = np.concatenate(data, axis=0)
            policy_ary, value_ary = self.model.model.predict_on_batch(array)
            idx = 0
            for conn, s in zip(ready_conns, size_list):
                conn.send((policy_ary[idx:idx+s], value_ary[idx:idx+s]))
                idx += s

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

    def try_reload_model(self):
        try:
            logger.debug("check model")
            if self.config.play.use_newest_next_generation_model:
                reload_newest_next_generation_model_if_changed(self.model, clear_session=True)
            else:
                reload_best_model_weight_if_changed(self.model, clear_session=True)
        except Exception as e:
            logger.error(e)


class MultiProcessReversiModelAPIClient(ReversiModelAPI):
    def __init__(self, config: Config, agent_model, conn):
        """

        :param config:
        :param reversi_zero.agent.model.ReversiModel agent_model:
        :param Connection conn:
        """
        super().__init__(config, agent_model)
        self.connection = conn

    def _do_predict(self, x):
        self.connection.send(x)
        return self.connection.recv()

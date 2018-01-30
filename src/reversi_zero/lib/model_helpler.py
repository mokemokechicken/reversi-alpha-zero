import os
from logging import getLogger
from time import sleep

import keras.backend as K


logger = getLogger(__name__)


def load_best_model_weight(model, clear_session=False):
    """

    :param reversi_zero.agent.model.ReversiModel model:
    :param bool clear_session:
    :return:
    """
    if clear_session:
        K.clear_session()
    return model.load(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path)


def save_as_best_model(model):
    """

    :param reversi_zero.agent.model.ReversiModel model:
    :return:
    """
    return model.save(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path)


def reload_best_model_weight_if_changed(model, clear_session=False):
    """

    :param reversi_zero.agent.model.ReversiModel model:
    :param bool clear_session:
    :return:
    """
    logger.debug(f"start reload the best model if changed")
    digest = model.fetch_digest(model.config.resource.model_best_weight_path)
    if digest != model.digest:
        return load_best_model_weight(model, clear_session=clear_session)

    logger.debug(f"the best model is not changed")
    return False


def reload_newest_next_generation_model_if_changed(model, clear_session=False):
    """

    :param reversi_zero.agent.model.ReversiModel model:
    :param bool clear_session:
    :return:
    """
    from reversi_zero.lib.data_helper import get_next_generation_model_dirs

    rc = model.config.resource
    dirs = get_next_generation_model_dirs(rc)
    if not dirs:
        logger.debug("No next generation model exists.")
        return False
    model_dir = dirs[-1]
    config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
    weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
    digest = model.fetch_digest(weight_path)
    if digest and digest != model.digest:
        logger.debug(f"Loading weight from {model_dir}")
        if clear_session:
            K.clear_session()
        for _ in range(5):
            try:
                return model.load(config_path, weight_path)
            except Exception as e:
                logger.warning(f"error in load model: #{e}")
                sleep(3)
        raise RuntimeError("Cannot Load Model!")

    else:
        logger.debug(f"The newest model is not changed: digest={digest}")
        return False

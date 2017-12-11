import os
from logging import getLogger


logger = getLogger(__name__)


def load_best_model_weight(model):
    """

    :param reversi_zero.agent.model.ReversiModel model:
    :return:
    """
    return model.load(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path)


def save_as_best_model(model):
    """

    :param reversi_zero.agent.model.ReversiModel model:
    :return:
    """
    return model.save(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path)


def reload_best_model_weight_if_changed(model):
    """

    :param reversi_zero.agent.model.ReversiModel model:
    :return:
    """
    logger.debug(f"start reload the best model if changed")
    digest = model.fetch_digest(model.config.resource.model_best_weight_path)
    if digest != model.digest:
        return load_best_model_weight(model)

    logger.debug(f"the best model is not changed")
    return False


def reload_newest_next_generation_model_if_changed(model):
    """

    :param reversi_zero.agent.model.ReversiModel model:
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
    if digest != model.digest:
        logger.debug(f"Loading weight from {model_dir}")
        return model.load(config_path, weight_path)
    else:
        return logger.debug("The newest model is not changed.")

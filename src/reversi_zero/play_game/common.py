from reversi_zero.config import Config
from reversi_zero.lib.model_helpler import reload_newest_next_generation_model_if_changed, load_best_model_weight


def load_model(config: Config):
    from reversi_zero.agent.model import ReversiModel
    model = ReversiModel(config)
    if config.play.use_newest_next_generation_model:
        loaded = reload_newest_next_generation_model_if_changed(model) or load_best_model_weight(model)
    else:
        loaded = load_best_model_weight(model) or reload_newest_next_generation_model_if_changed(model)
    if not loaded:
        raise RuntimeError("No models found!")
    return model

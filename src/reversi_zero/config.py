import os


def _project_dir():
    d = os.path.dirname
    return d(d(d(os.path.abspath(__file__))))


def _data_dir():
    return os.path.join(_project_dir(), "data")


class Config:
    def __init__(self):
        self.opts = Options()
        self.resource = ResourceConfig()
        self.model = SmallModelConfig()
        self.play = PlayConfig()
        self.play_data = PlayDataConfig()
        self.trainer = TrainerConfig()
        self.eval = EvaluateConfig()


class Options:
    new = False


class EvaluateConfig:
    def __init__(self):
        self.game_num = 400
        self.replace_rate = 0.55
        self.play_config = PlayConfig()
        self.play_config.change_tau_turn = 0
        self.play_config.noise_eps = 0


class PlayDataConfig:
    def __init__(self):
        self.nb_game_in_file = 100
        self.max_file_num = 10


class PlayConfig:
    def __init__(self):
        self.simulation_num_per_move = 10
        self.c_puct = 1
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.03
        self.change_tau_turn = 10


class TrainerConfig:
    def __init__(self):
        self.batch_size = 2048
        self.epoch_to_checkpoint = 1
        self.start_total_steps = 0


class ResourceConfig:
    def __init__(self):
        self.project_dir = os.environ.get("PROJECT_DIR", _project_dir())
        self.data_dir = os.environ.get("DATA_DIR", _data_dir())
        self.model_dir = os.environ.get("MODEL_DIR", os.path.join(self.data_dir, "model"))
        self.model_best_config_path = os.path.join(self.model_dir, "model_best_config.json")
        self.model_best_weight_path = os.path.join(self.model_dir, "model_best_weight.h5")

        self.next_generation_model_dir = os.path.join(self.model_dir, "next_generation")
        self.next_generation_model_dirname_tmpl = "model_%s"
        self.next_generation_model_config_filename = "model_config.json"
        self.next_generation_model_weight_filename = "model_weight.h5"

        self.play_data_dir = os.path.join(self.data_dir, "play_data")
        self.play_data_filename_tmpl = "play_%s.json"

        self.log_dir = os.path.join(self.project_dir, "logs")
        self.main_log_path = os.path.join(self.log_dir, "main.log")

    def create_directories(self):
        dirs = [self.project_dir, self.data_dir, self.model_dir, self.play_data_dir, self.log_dir,
                self.next_generation_model_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)


class SmallModelConfig:
    cnn_filter_num = 16
    cnn_filter_size = 3
    res_layer_num = 1
    l2_reg = 1e-4
    value_fc_size = 16


class ModelConfig:
    cnn_filter_num = 256
    cnn_filter_size = 3
    res_layer_num = 10
    l2_reg = 1e-4
    value_fc_size = 256


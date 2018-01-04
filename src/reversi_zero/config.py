import os


def _project_dir():
    d = os.path.dirname
    return d(d(d(os.path.abspath(__file__))))


def _data_dir():
    return os.path.join(_project_dir(), "data")


class Config:
    def __init__(self, config_type="mini"):
        self.opts = Options()
        self.resource = ResourceConfig()
        self.gui = GuiConfig()

        if config_type == "mini":
            import reversi_zero.configs.mini as c
        elif config_type == "normal":
            import reversi_zero.configs.normal as c
        else:
            raise RuntimeError(f"unknown config_type: {config_type}")
        self.model = c.ModelConfig()
        self.play = c.PlayConfig()
        self.play_data = c.PlayDataConfig()
        self.trainer = c.TrainerConfig()
        self.eval = c.EvaluateConfig()


class Options:
    new = False


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
        self.tensorboard_log_dir = os.path.join(self.log_dir, 'tensorboard')

    def create_directories(self):
        dirs = [self.project_dir, self.data_dir, self.model_dir, self.play_data_dir, self.log_dir,
                self.next_generation_model_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)


class GuiConfig:
    def __init__(self):
        self.window_size = (400, 440)
        self.window_title = "reversi-alpha-zero"


class PlayWithHumanConfig:
    def __init__(self):
        self.simulation_num_per_move = 400
        self.thinking_loop = 1
        self.logging_thinking = True
        self.c_puct = 1
        self.parallel_search_num = 8
        self.noise_eps = 0
        self.change_tau_turn = 0
        self.resign_threshold = None
        self.use_newest_next_generation_model = True

    def update_play_config(self, pc):
        """

        :param reversi_zero.configs.normal.PlayConfig pc:
        :return:
        """
        pc.simulation_num_per_move = self.simulation_num_per_move
        pc.thinking_loop = self.thinking_loop
        pc.logging_thinking = self.logging_thinking
        pc.c_puct = self.c_puct
        pc.noise_eps = self.noise_eps
        pc.change_tau_turn = self.change_tau_turn
        pc.parallel_search_num = self.parallel_search_num
        pc.resign_threshold = self.resign_threshold
        pc.use_newest_next_generation_model = self.use_newest_next_generation_model

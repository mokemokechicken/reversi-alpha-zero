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
        self.max_file_num = 5000


class PlayConfig:
    def __init__(self):
        self.simulation_num_per_move = 50
        self.c_puct = 1
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.03
        self.change_tau_turn = 10


class TrainerConfig:
    def __init__(self):
        self.batch_size = 2048
        self.epoch_to_checkpoint = 1
        self.start_total_steps = 0


class ModelConfig:
    cnn_filter_num = 256
    cnn_filter_size = 3
    res_layer_num = 10
    l2_reg = 1e-4
    value_fc_size = 256

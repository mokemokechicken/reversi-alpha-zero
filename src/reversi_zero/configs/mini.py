class EvaluateConfig:
    def __init__(self):
        self.game_num = 100
        self.replace_rate = 0.55
        self.play_config = PlayConfig()
        self.play_config.c_puct = 1
        self.play_config.change_tau_turn = 0
        self.play_config.noise_eps = 0
        self.evaluate_latest_first = True


class PlayDataConfig:
    def __init__(self):
        self.nb_game_in_file = 100
        self.max_file_num = 10
        self.save_policy_of_tau_1 = True


class PlayConfig:
    def __init__(self):
        self.simulation_num_per_move = 10
        self.thinking_loop = 1
        self.logging_thinking = False
        self.c_puct = 5
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.5
        self.change_tau_turn = 10
        self.virtual_loss = 3
        self.prediction_queue_size = 16
        self.parallel_search_num = 4
        self.prediction_worker_sleep_sec  = 0.00001
        self.wait_for_expanding_sleep_sec = 0.000001
        self.resign_threshold = -0.8
        self.disable_resignation_rate = 0.1
        self.false_positive_threshold = 0.05
        self.resign_threshold_delta = 0.01
        self.use_newest_next_generation_model = True


class TrainerConfig:
    def __init__(self):
        self.batch_size = 2048
        self.min_data_size_to_learn = 100
        self.epoch_to_checkpoint = 1
        self.start_total_steps = 0
        self.save_model_steps = 200


class ModelConfig:
    cnn_filter_num = 16
    cnn_filter_size = 3
    res_layer_num = 1
    l2_reg = 1e-4
    value_fc_size = 16

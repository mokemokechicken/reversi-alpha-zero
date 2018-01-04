class EvaluateConfig:
    def __init__(self):
        self.game_num = 200  # 400
        self.replace_rate = 0.55
        self.play_config = PlayConfig()
        self.play_config.simulation_num_per_move = 400
        self.play_config.thinking_loop = 1
        self.play_config.change_tau_turn = 0
        self.play_config.noise_eps = 0
        self.play_config.disable_resignation_rate = 0
        self.evaluate_latest_first = True


class PlayDataConfig:
    def __init__(self):
        # Max Training Data Size = nb_game_in_file * max_file_num * 8
        self.nb_game_in_file = 5
        self.max_file_num = 300
        self.save_policy_of_tau_1 = True


class PlayConfig:
    def __init__(self):
        self.simulation_num_per_move = 400
        self.thinking_loop = 1
        self.logging_thinking = False
        self.c_puct = 1
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.5
        self.change_tau_turn = 10
        self.virtual_loss = 3
        self.prediction_queue_size = 16
        self.parallel_search_num = 8
        self.prediction_worker_sleep_sec  = 0.0001
        self.wait_for_expanding_sleep_sec = 0.00001
        self.resign_threshold = -0.9
        self.disable_resignation_rate = 0.1
        self.false_positive_threshold = 0.05
        self.resign_threshold_delta = 0.01

        # True means evaluating 'AlphaZero' method (disable 'eval' worker).
        # Please change to False if you want to evaluate 'AlphaGo Zero' method.
        self.use_newest_next_generation_model = True


class TrainerConfig:
    def __init__(self):
        self.batch_size = 512  # 2048
        self.min_data_size_to_learn = 100000
        self.epoch_to_checkpoint = 1
        self.start_total_steps = 0
        self.save_model_steps = 200
        self.use_tensorboard = True
        self.logging_per_steps = 100


class ModelConfig:
    cnn_filter_num = 256
    cnn_filter_size = 3
    res_layer_num = 10
    l2_reg = 1e-4
    value_fc_size = 256

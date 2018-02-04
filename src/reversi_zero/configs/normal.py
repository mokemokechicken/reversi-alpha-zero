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
        self.multi_process_num = 16
        self.nb_game_in_file = 2
        self.max_file_num = 800
        self.save_policy_of_tau_1 = True
        self.enable_ggf_data = True
        self.nb_game_in_ggf_file = 100


class PlayConfig:
    def __init__(self):
        self.simulation_num_per_move = 50
        self.share_mtcs_info_in_self_play = True
        self.reset_mtcs_info_per_game = 5
        self.thinking_loop = 10
        self.required_visit_to_decide_action = 400
        self.start_rethinking_turn = 8
        self.c_puct = 1
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.5
        self.dirichlet_noise_only_for_legal_moves = True
        self.change_tau_turn = 3
        self.virtual_loss = 3
        self.prediction_queue_size = 16
        self.parallel_search_num = 8
        self.prediction_worker_sleep_sec  = 0.0001
        self.wait_for_expanding_sleep_sec = 0.00001
        self.resign_threshold = -0.9
        self.allowed_resign_turn = 20
        self.disable_resignation_rate = 0.1
        self.false_positive_threshold = 0.05
        self.resign_threshold_delta = 0.01
        self.policy_decay_turn = 60
        self.policy_decay_power = 1

        #
        self.schedule_of_simulation_num_per_move = [
            (0, 8),
            (300, 50),
            (2000, 200),
        ]

        # True means evaluating 'AlphaZero' method (disable 'eval' worker).
        # Please change to False if you want to evaluate 'AlphaGo Zero' method.
        self.use_newest_next_generation_model = True


class TrainerConfig:
    def __init__(self):
        self.wait_after_save_model_ratio = 1  # wait after saving model
        self.batch_size = 256  # 2048
        self.min_data_size_to_learn = 100000
        self.epoch_to_checkpoint = 1
        self.start_total_steps = 0
        self.save_model_steps = 200
        self.use_tensorboard = True
        self.logging_per_steps = 100
        self.lr_schedules = [
            (0, 0.01),
            (150000, 0.001),
            (300000, 0.0001),
        ]


class ModelConfig:
    cnn_filter_num = 256
    cnn_filter_size = 3
    res_layer_num = 10
    l2_reg = 1e-4
    value_fc_size = 256

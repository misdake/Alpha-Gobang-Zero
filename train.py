# coding: utf-8
from alphazero.train import TrainModel


train_config = {
    'lr': 1e-2,
    'c_puct': 3,
    'board_len': 5,
    'batch_size': 1000,
    'is_use_gpu': True,
    'n_test_games': 10,
    'n_mcts_iters': 100,
    'n_self_plays': 4000,
    'is_save_game': True,
    'n_feature_planes': 2,
    'check_frequency': 100,
    'start_train_size': 1000
}
train_model = TrainModel(**train_config)
train_model.train()

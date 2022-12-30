# coding: utf-8
from alphazero.train import TrainModel, ValueType

train_config_A = {
    'name': 'A',
    'board_len': 5,
    'n_mcts_iters': 200,
    'n_feature_planes': 2,
    'value_type': ValueType.WinDrawLose
}

train_config_B = {
    'name': 'B',
    'board_len': 5,
    'n_mcts_iters': 100,
    'n_feature_planes': 2,
    'value_type': ValueType.BubbleCount
}

train_config_C = {
    'name': 'C',
    'board_len': 5,
    'n_mcts_iters': 200,
    'n_feature_planes': 2,
    'value_type': ValueType.BubbleCount
}

train_config_D = {
    'name': 'D',
    'board_len': 5,
    'n_mcts_iters': 200,
    'n_feature_planes': 2,
    'value_type': ValueType.Combined
}

train_config_E = {
    'name': 'E',
    'board_len': 5,
    'n_mcts_iters': 200,
    'n_feature_planes': 6,
    'value_type': ValueType.Combined
}


train_model = TrainModel(**train_config_B)
train_model.train()

# coding: utf-8
from alphazero.train import TrainModel, ValueType

config_A = {
    'name': 'A',
    'model': 'A/A_500.pth',
    'board_len': 5,
    'n_mcts_iters': 200,
    'n_feature_planes': 2,
    'value_type': ValueType.WinDrawLose
}

config_B = {
    'name': 'B',
    'model': 'B/B_200.pth',
    'board_len': 5,
    'n_mcts_iters': 100,
    'n_feature_planes': 2,
    'value_type': ValueType.BubbleCount
}

config_C = {
    'name': 'C',
    'model': 'C/C_200.pth',
    'board_len': 5,
    'n_mcts_iters': 200,
    'n_feature_planes': 2,
    'value_type': ValueType.BubbleCount
}

config_D = {
    'name': 'D',
    'model': 'D/D_200.pth',
    'board_len': 5,
    'n_mcts_iters': 200,
    'n_feature_planes': 2,
    'value_type': ValueType.Combined
}

config_E = {
    'name': 'E',
    'model': 'E/E_200.pth',
    'board_len': 5,
    'n_mcts_iters': 200,
    'n_feature_planes': 6,
    'value_type': ValueType.Combined
}


train_model = TrainModel(**config_C)
train_model.train()

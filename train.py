# coding: utf-8
from dataclasses import dataclass

from alphazero.train import TrainModel, ValueType


@dataclass
class AgentConfig:
    def __init__(self, name: str, model: str, n_mcts_iters: int, n_feature_planes: int, value_type: ValueType):
        self.name = name
        self.model = model
        self.board_len = 5
        self.n_mcts_iters = n_mcts_iters
        self.n_feature_planes = n_feature_planes
        self.value_type = value_type


config_A = AgentConfig('A', 'A/A_500', 200, 2, ValueType.WinDrawLose)
config_B = AgentConfig('B', 'B/B_200', 100, 2, ValueType.BubbleCount)
config_C = AgentConfig('C', 'C/C_200', 200, 2, ValueType.BubbleCount)
config_D = AgentConfig('D', 'D/D_200', 200, 6, ValueType.BubbleCount)
config_E = AgentConfig('E', 'E/E_200', 200, 6, ValueType.Combined)

if __name__ == '__main__':
    config = config_C

    train_model = TrainModel(config.name, config.board_len, config.n_mcts_iters, config.n_feature_planes,
                             config.value_type)
    train_model.train()

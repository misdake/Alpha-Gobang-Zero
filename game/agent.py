# coding:utf-8
import torch

from alphazero import AlphaZeroMCTS, PolicyValueNet, BubbleBoard


def test_model(model: str):
    """ 测试模型是否可用

    Parameters
    ----------
    model: str
        模型路径
    """
    try:
        model = torch.load(model, map_location=torch.device('cpu'))
        return isinstance(model, PolicyValueNet)
    except:
        return False


class Agent:
    """ AI """

    def __init__(self, bubble_board: BubbleBoard, model: str, n_iters):
        self.bubble_board = bubble_board
        self.device = torch.device('cpu')
        if model and test_model(model):
            self.model = torch.load(model, map_location=torch.device('cpu')).to(self.device)  # type:PolicyValueNet
            self.model.set_device()
            self.model.eval()
            self.mcts = AlphaZeroMCTS(self.model, 3, n_iters)
        else:
            print('model not loaded')
            exit(-1)

    def run(self) -> int:
        """ 根据当前局面获取动作 """
        action = self.mcts.get_action(self.bubble_board)
        return action

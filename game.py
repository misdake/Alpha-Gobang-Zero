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

    def __init__(self, bubble_board: BubbleBoard, model: str, c_puct=5.0, n_iters=2000):
        """
        Parameters
        ----------
        bubble_board: BubbleBoard
            棋盘

        model: str
            模型路径

        c_puct: float
            探索常数

        n_iters: int
            蒙特卡洛树搜索次数
        """
        self.bubble_board = bubble_board
        self.c_puct = c_puct
        self.n_iters = n_iters
        self.device = torch.device('cpu')
        if model and test_model(model):
            print('model loaded')
            self.model = torch.load(model, map_location=torch.device('cpu')).to(self.device)  # type:PolicyValueNet
            self.model.set_device()
            self.model.eval()
            self.mcts = AlphaZeroMCTS(self.model, c_puct, n_iters)
        else:
            print('model not loaded')
            exit(-1)

    def run(self) -> int:
        """ 根据当前局面获取动作 """
        action = self.mcts.get_action(self.bubble_board)
        return action


board = BubbleBoard(board_len=5)
agent = Agent(board, "model/best_policy_value_net.pth")

board.print()

self_play = False
playing = True
while playing:
    if self_play:
        input_var = "ai"
    else:
        input_var = input("Your turn : ")
        if input_var == "exit":
            playing = False
            break

    if input_var == "self_play":
        self_play = True

    if input_var == "ai" or self_play:
        print('ai running...')
        action = agent.run()
        x = action // 5
        y = action % 5
    else:
        split = input_var.split(",")
        try:
            if len(split) == 2:
                x = int(split[0])
                y = int(split[1])
            else:
                print("cannot parse, type x,y")
                continue
        except ValueError:
            print("cannot parse, type x,y")
            continue

    valid = board.do_action_with_check((x, y))
    if not valid:
        print("invalid position")
        continue

    print(f'player position: {x}, {y}')
    board.print((x, y))

    is_over, _ = board.is_game_over()
    if is_over:
        print('You win!')
        print()
        board.clear_board()
        board.print()
        continue

    print('ai running...')
    action = agent.run()
    board.do_action(action)
    x = action // 5
    y = action % 5

    print(f'ai position: {x}, {y}')
    board.print((x, y))

    is_over, _ = board.is_game_over()
    if is_over:
        print('You lose!')
        print()
        board.clear_board()
        board.print()
        continue

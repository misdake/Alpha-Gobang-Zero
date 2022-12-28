# coding:utf-8
import json
import math
import os
import time
import traceback

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from .alpha_zero_mcts import AlphaZeroMCTS
from .bubble_board import BubbleBoard
from .policy_value_net import PolicyValueNet
from .self_play_dataset import SelfPlayData, SelfPlayDataSet


def exception_handler(train_func):
    """ 异常处理装饰器 """

    def wrapper(train_pipe_line, *args, **kwargs):
        try:
            train_func(train_pipe_line)
        except BaseException as e:
            if not isinstance(e, KeyboardInterrupt):
                traceback.print_exc()

            t = time.strftime('%Y-%m-%d_%H-%M-%S',
                              time.localtime(time.time()))
            train_pipe_line.save_model(
                f'last_policy_value_net_{t}', 'train_losses', 'games')

    return wrapper


class PolicyValueLoss(nn.Module):
    """ 根据 self-play 产生的 `z` 和 `π` 计算误差 """

    def __init__(self):
        super().__init__()

    def forward(self, p_hat, pi, value, z):
        """ 前馈

        Parameters
        ----------
        p_hat: Tensor of shape (N, board_len^2)
            对数动作概率向量

        pi: Tensor of shape (N, board_len^2)
            `mcts` 产生的动作概率向量

        value: Tensor of shape (N, )
            对每个局面的估值

        z: Tensor of shape (N, )
            最终的游戏结果相对每一个玩家的奖赏
        """
        value_loss = F.mse_loss(value, z)
        policy_loss = -torch.sum(pi * p_hat, dim=1).mean()
        loss = value_loss + policy_loss
        return loss


class TrainModel:
    """ 训练模型 """

    def __init__(self, board_len=5, lr=0.01, n_self_plays=1500, n_mcts_iters=500,
                 n_feature_planes=2, batch_size=500, start_train_size=500, check_frequency=100,
                 n_test_games=10, c_puct=4, is_save_game=False, **kwargs):
        """
        Parameters
        ----------
        board_len: int
            棋盘大小

        lr: float
            学习率

        n_self_plays: int
            自我博弈游戏局数

        n_mcts_iters: int
            蒙特卡洛树搜索次数

        n_feature_planes: int
            特征平面个数

        batch_size: int
            mini-batch 的大小

        start_train_size: int
            开始训练模型时的最小数据集尺寸

        check_frequency: int
            测试模型的频率

        n_test_games: int
            测试模型时与历史最优模型的比赛局数

        c_puct: float
            探索常数

        is_use_gpu: bool
            是否使用 GPU

        is_save_game: bool
            是否保存自对弈的棋谱
        """
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.n_self_plays = n_self_plays
        self.n_test_games = n_test_games
        self.n_mcts_iters = n_mcts_iters
        self.is_save_game = is_save_game
        self.check_frequency = check_frequency
        self.start_train_size = start_train_size
        self.device = torch.device('cpu')
        self.bubble_board = BubbleBoard(board_len, n_feature_planes)

        # 创建策略-价值网络和蒙特卡洛搜索树
        self.policy_value_net = self.__get_policy_value_net(board_len)
        self.mcts = AlphaZeroMCTS(self.policy_value_net, c_puct=c_puct, n_iters=n_mcts_iters, is_self_play=True)

        # 创建优化器和损失函数
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = PolicyValueLoss()
        self.lr_scheduler = MultiStepLR(self.optimizer, [100, 200, 300], gamma=0.1)

        # 创建数据集
        self.dataset = SelfPlayDataSet(board_len)

        # 记录数据
        self.train_losses = self.__load_data('log/train_losses.json')
        self.games = self.__load_data('log/games.json')

    def __self_play(self):
        """ 自我博弈一局

        Returns
        -------
        self_play_data: namedtuple
            自我博弈数据，有以下三个成员:
            * `pi_list`: 蒙特卡洛树搜索产生的动作概率向量 π 组成的列表
            * `z_list`: 一局之中每个动作的玩家相对最后的游戏结果的奖赏列表
            * `feature_planes_list`: 一局之中每个动作对应的特征平面组成的列表
        """
        # 初始化棋盘和数据容器
        self.policy_value_net.eval()
        board = self.bubble_board
        board.clear_board()
        pi_list, feature_planes_list, players = [], [], []
        action_list, z_list = [], []

        # 开始一局游戏
        while True:
            player = board.current_player
            # curr_reward = board.get_state_reward(player)
            action, pi = self.mcts.get_action(board)

            # board.print((action // self.bubble_board.board_len, action % self.bubble_board.board_len))

            # 保存每一步的数据
            feature_planes_list.append(board.get_feature_planes())
            players.append(player)
            action_list.append(action)
            pi_list.append(pi)
            board.do_action(action)

            # 判断游戏是否结束
            is_over, winner = board.is_game_over_with_limit()

            # 记录状态价值
            next_reward = board.get_state_reward(player)
            # action_reward = next_reward - curr_reward

            z_list.append(math.tanh(next_reward))

            if player > 0:
                print('+', end='')
            else:
                print('-', end='')
            print(f'{action}({next_reward:.3}) ', end='')
            if board.action_len % 10 == 0:
                print()

            if is_over:
                print()
                break

        # 重置根节点
        self.mcts.reset_root()

        # 返回数据
        if self.is_save_game:
            self.games.append(action_list)

        self_play_data = SelfPlayData(
            pi_list=pi_list, z_list=z_list, feature_planes_list=feature_planes_list)
        return self_play_data

    @exception_handler
    def train(self):
        train_count = 0
        """ 训练模型 """
        for i in range(self.n_self_plays):
            print(f'🏹 正在进行第 {i + 1} 局自我博弈游戏...')
            self.dataset.append(self.__self_play())

            # 如果数据集中的数据量大于 start_train_size 就进行一次训练
            if len(self.dataset) >= self.start_train_size:
                data_loader = iter(DataLoader(self.dataset, self.batch_size, shuffle=True, drop_last=False))
                print(f'💊 第 {train_count + 1} 次训练...')
                train_count += 1

                self.policy_value_net.train()
                # 随机选出一批数据来训练，防止过拟合
                feature_planes, pi, z = next(data_loader)
                feature_planes = feature_planes.to(self.device)
                pi, z = pi.to(self.device), z.to(self.device)

                # 前馈
                p_hat, value = self.policy_value_net(feature_planes)
                # 梯度清零
                self.optimizer.zero_grad()
                # 计算损失
                loss = self.criterion(p_hat, pi, value.flatten(), z)
                # 误差反向传播
                loss.backward()
                # 更新参数
                self.optimizer.step()
                # 学习率退火
                self.lr_scheduler.step()

                # 记录误差
                self.train_losses.append([i, loss.item()])
                print(f"🚩 train_loss = {loss.item():<10.5f}\n")

                if train_count % 50 == 0:
                    model_path = f'model/checkpoint/saved_bubble_reward_{train_count}.pth'
                    torch.save(self.mcts.policy_value_net, model_path)
            # 测试模型
            # if (i + 1) % self.check_frequency == 0:
            #     self.__test_model()

    def save_model(self, model_name: str, loss_name: str, game_name: str):
        """ 保存模型

        Parameters
        ----------
        model_name: str
            模型文件名称，不包含后缀

        loss_name: str
            损失文件名称，不包含后缀

        game_name: str
            自对弈棋谱名称，不包含后缀
        """
        os.makedirs('model', exist_ok=True)

        path = f'model/{model_name}.pth'
        self.policy_value_net.eval()
        torch.save(self.policy_value_net, path)
        print(f'🎉 已将当前模型保存到 {os.path.join(os.getcwd(), path)}')

        # 保存数据
        with open(f'log/{loss_name}.json', 'w', encoding='utf-8') as f:
            json.dump(self.train_losses, f)

        if self.is_save_game:
            with open(f'log/{game_name}.json', 'w', encoding='utf-8') as f:
                json.dump(self.games, f)

    def __do_mcts_action(self, mcts):
        """ 获取动作 """
        action = mcts.get_action(self.bubble_board)
        self.bubble_board.do_action(action)
        is_over, winner = self.bubble_board.is_game_over()
        return is_over, winner

    def __get_policy_value_net(self, board_len=9):
        """ 创建策略-价值网络，如果存在历史最优模型则直接载入最优模型 """
        os.makedirs('model', exist_ok=True)

        best_model = 'best_policy_value_net.pth'
        history_models = sorted(
            [i for i in os.listdir('model') if i.startswith('last')])

        # 从历史模型中选取最新模型
        model = history_models[-1] if history_models else best_model
        model = f'model/{model}'
        if os.path.exists(model):
            print(f'💎 载入模型 {model} ...\n')
            net = torch.load(model, map_location=torch.device('cpu')).to(self.device)  # type:PolicyValueNet
        else:
            print(f'💎 初始化模型 {model} ...\n')
            net = PolicyValueNet(n_feature_planes=self.bubble_board.n_feature_planes,
                                 board_len=board_len).to(self.device)

        return net

    def __load_data(self, path: str):
        """ 载入历史损失数据 """
        data = []
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except:
            os.makedirs('log', exist_ok=True)

        return data

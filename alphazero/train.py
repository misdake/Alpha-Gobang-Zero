# coding:utf-8
import json
import math
import os
import time
import traceback
import random

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from .alpha_zero_mcts import AlphaZeroMCTS
from .bubble_board import BubbleBoard
from .policy_value_net import PolicyValueNet
from .self_play_dataset import SelfPlayData, SelfPlayDataSet

from enum import Enum


class ValueType(Enum):
    WinDrawLose = 1
    BubbleCount = 2
    Combined = 3


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

    def __init__(self, name, board_w: int, board_h: int, n_mcts_iters, n_feature_planes, value_type=ValueType.BubbleCount):
        self.name = name
        self.c_puct = 3
        self.batch_size = 1000
        self.n_self_plays = 10000
        self.n_mcts_iters = n_mcts_iters
        self.is_save_game = True
        self.start_train_size = 2000
        self.value_type = value_type

        self.device = torch.device('cpu')
        self.bubble_board = BubbleBoard(board_w, board_h, n_feature_planes)

        # 创建策略-价值网络和蒙特卡洛搜索树
        self.policy_value_net = self.__get_policy_value_net(board_w, board_h)
        self.mcts = AlphaZeroMCTS(self.policy_value_net, c_puct=self.c_puct, n_iters=n_mcts_iters, is_self_play=True)

        # 创建优化器和损失函数
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=0.01, weight_decay=1e-4)
        self.loss = PolicyValueLoss()
        self.lr_scheduler = MultiStepLR(self.optimizer, [100, 300, 500], gamma=0.1)

        # 创建数据集
        self.dataset = SelfPlayDataSet(board_w, board_h)

        # 记录数据
        self.train_losses = self.__load_data(f'log/train_losses_{self.name}.json')
        self.games = self.__load_data('log/games.json')

    def __self_play(self, train_iter):
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

        # 50%概率随机一个开局，防止自对弈情况下开局形成定式
        if random.randint(0, 1) == 1:
            for _ in range(random.randint(0, 100)):
                choice = random.choice(board.available_actions)
                board.do_action(choice)

        pi_list, feature_planes_list, players = [], [], []
        action_list, z_list = [], []

        # 开始一局游戏
        while True:
            player = board.current_player
            curr_reward = board.get_state_reward(player)
            action, pi = self.mcts.get_action(board)

            if train_iter % 10 == 0:
                print()
                board.print((action % self.bubble_board.board_w, action // self.bubble_board.board_w))

            # 保存每一步的数据
            feature_plane = board.get_feature_planes(player)
            feature_planes_list.append(feature_plane)
            players.append(player)
            action_list.append(action)
            pi_list.append(pi)
            board.do_action(action)

            # 判断游戏是否结束
            is_over, winner = board.is_game_over_with_limit()

            # 记录状态价值
            # next_reward = board.get_state_reward(player)
            # action_reward = next_reward - curr_reward

            value = math.tanh(curr_reward)
            z_list.append(value)

            if player > 0:
                print('+', end='')
            else:
                print('-', end='')
            print(f'{action:02}({value:.3}) ', end='')
            if board.action_len % 10 == 0:
                print(f'  --{board.action_len}')

            if is_over:
                print()

                if self.value_type == ValueType.Combined:
                    if winner != 0:
                        wdl = [1 if i == winner else -1 for i in players]
                    else:
                        wdl = [0] * len(players)
                    state_ratio = 0.98 ** train_iter  # 0.66@20, 0.36@50, 0.13@100
                    wdl_ratio = 1.0 - state_ratio
                    z_list = [state_ratio * z_list[i] + wdl_ratio * wdl[i] for i in range(len(players))]

                if self.value_type == ValueType.WinDrawLose:
                    # 改用传统棋类的方法设置value
                    if winner != 0:
                        z_list = [1 if i == winner else -1 for i in players]
                    else:
                        z_list = [0] * len(players)

                print(f'winner {winner}')

                break

        # 重置根节点
        self.mcts.reset_root()

        # 返回数据
        if self.is_save_game:
            self.games.append(action_list)

        self_play_data = SelfPlayData(pi_list=pi_list, z_list=z_list, feature_planes_list=feature_planes_list)
        return self_play_data

    @exception_handler
    def train(self):
        train_iter = 0
        """ 训练模型 """
        for i in range(self.n_self_plays):
            print(f'🏹 正在进行第 {i + 1} 局自我博弈游戏...')
            self.dataset.append(self.__self_play(train_iter))

            print(f'数据集容量 {len(self.dataset)}')

            # 如果数据集中的数据量大于 start_train_size 就进行一次训练
            if len(self.dataset) >= self.start_train_size:
                for _ in range(2):
                    data_loader = iter(DataLoader(self.dataset, self.batch_size, shuffle=True, drop_last=False))
                    print(f'💊 第 {train_iter + 1} 次训练...')
                    train_iter += 1

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
                    loss = self.loss(p_hat, pi, value.flatten(), z)
                    # 误差反向传播
                    loss.backward()
                    # 更新参数
                    self.optimizer.step()
                    # 学习率退火
                    self.lr_scheduler.step()

                    # 记录误差
                    self.train_losses.append([i, loss.item()])
                    print(f"🚩 train_loss = {loss.item():<10.5f}\n")

                    if train_iter % 50 == 0:
                        model_path = f'model/checkpoint/{self.name}_{train_iter}.pth'
                        torch.save(self.mcts.policy_value_net, model_path)

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

    def __get_policy_value_net(self, board_w: int, board_h: int):
        """ 创建策略-价值网络，如果存在历史最优模型则直接载入最优模型 """
        os.makedirs('checkpoint', exist_ok=True)
        os.makedirs('history', exist_ok=True)

        best_model = 'best_policy_value_net.pth'
        history_models = sorted([i for i in os.listdir('model') if i.startswith('last')])

        # 从历史模型中选取最新模型
        model = history_models[-1] if history_models else best_model
        model = f'model/{model}'
        if os.path.exists(model):
            print(f'💎 载入模型 {model} ...\n')
            net = torch.load(model, map_location=torch.device('cpu')).to(self.device)  # type:PolicyValueNet
        else:
            print(f'💎 初始化模型 {model} ...\n')
            net = PolicyValueNet(n_feature_planes=self.bubble_board.n_feature_planes,
                                 board_w=board_w, board_h=board_h).to(self.device)

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

# coding: utf-8
from typing import Tuple
from copy import deepcopy
from collections import OrderedDict

import torch
import numpy as np


class BubbleBoard:
    """ 棋盘类 """

    WHITE = -1
    EMPTY = 0
    BLACK = 1

    def __init__(self, board_w: int, board_h: int, n_feature_planes=2):
        self.board_w = board_w
        self.board_h = board_h
        self.cell_len = self.board_w * self.board_h
        self.n_feature_planes = n_feature_planes

        # 棋盘状态字典，key 为 action，value 为 current_player
        self.state = OrderedDict()

        # 初始化一下，不然python不开心
        self.black_count = 1
        self.white_count = 1
        self.action_len = 0
        self.current_player = self.BLACK
        self.black_available_points = []
        self.white_available_points = []
        self.available_actions = self.black_available_points

        if self.n_feature_planes > 2:
            self.state_history = [self.state.copy()] * (self.n_feature_planes // 2)
        else:
            self.state_history = []

        # 重复的调用
        self.clear_board()

    def copy(self):
        """ 复制棋盘 """
        return deepcopy(self)

    def clear_board(self):
        """ 清空棋盘 """
        self.action_len = 0
        self.state.clear()
        self.current_player = self.BLACK
        self.black_count = 1
        self.white_count = 1
        self.available_actions = self.black_available_points

        self.state[0] = self.BLACK
        self.state[self.cell_len - 1] = self.WHITE
        self._refresh_available_points()

        if self.n_feature_planes > 2:
            self.state_history = [self.state.copy()] * (self.n_feature_planes // 2)

    def do_action(self, action: int):
        """ 落子并更新棋盘

        Parameters
        ----------
        action: int
            落子位置，范围为 `[0, board_len^2 -1]`
        """

        self.action_len += 1
        if self.current_player == self.WHITE:
            self.white_count += 1
        if self.current_player == self.BLACK:
            self.black_count += 1

        expand = False
        if self.state.get(action, self.EMPTY) == self.EMPTY:  # 空位置
            self.state[action] = self.current_player  # 设置泡泡的量为1
            if self.current_player == self.WHITE:
                self.black_available_points.remove(action)
            if self.current_player == self.BLACK:
                self.white_available_points.remove(action)
        else:
            if np.sign(self.state[action]) != self.current_player:
                print('!!!!!!!!!!!!!!!!!')
            self.state[action] += np.sign(self.state[action])  # 加一个泡泡，经过外层available处理一定是合法的
            if abs(self.state[action]) == 4:
                expand = True  # 触发分裂

        # 如果存在分裂，就要大动干戈了，可能会反复传播。遍历当前列表，传播的放到下个列表，然后跑下个列表，重复直到安静
        if expand:
            curr_iter = set()
            curr_iter.add(action)
            next_iter = set()
            while len(curr_iter) > 0:
                # 对于每个curr，都将4个泡泡分裂到四周
                for i in iter(curr_iter):
                    x = i % self.board_w
                    y = i // self.board_w
                    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
                    i_player = np.sign(self.state[i])  # 玩家的符号位

                    self.state[i] -= i_player * 4  # 4个泡泡分裂开，我想了一下应该不可能到8的吧，所以只减4就行
                    for d in range(4):  # 向四个方向分裂开
                        x_t = x + directions[d][0]
                        y_t = y + directions[d][1]
                        if 0 <= y_t < self.board_h and 0 <= x_t < self.board_w:
                            j = y_t * self.board_w + x_t  # 上下左右中的一个
                            self.state[j] = i_player * (abs(self.state.get(j, self.EMPTY)) + 1)  # 加一个泡泡，并设置玩家
                            if abs(self.state[j]) == 4:  # 如果到达4个就要下个轮回新增
                                next_iter.add(j)
                # curr都计算结束，进入下一个计算循环
                curr_iter = next_iter
                next_iter = set()

            # 循环结束，重新计算available_points
            self._refresh_available_points()

        # 交换玩家
        if self.current_player == self.BLACK:
            self.available_actions = self.white_available_points
            self.current_player = self.WHITE
        elif self.current_player == self.WHITE:
            self.available_actions = self.black_available_points
            self.current_player = self.BLACK
        else:
            print("?")

        if self.n_feature_planes > 2:
            self.state_history.pop(0)
            self.state_history.append(self.state.copy())

    def _refresh_available_points(self):
        self.black_available_points.clear()
        self.white_available_points.clear()
        self.black_count = 0
        self.white_count = 0
        for i in range(self.cell_len):
            state = self.state.get(i, self.EMPTY)
            if state >= 0:
                self.black_count += abs(state)
                self.black_available_points.append(i)
            if state <= 0:
                self.white_count += abs(state)
                self.white_available_points.append(i)

    def do_action_with_check(self, pos: tuple) -> bool:
        """ 落子并更新棋盘，只提供给 app 使用

        Parameters
        ----------
        pos: Tuple[int, int]
            落子在棋盘上的位置，范围为 `(0, 0) ~ (board_len-1, board_len-1)`

        Returns
        -------
        update_ok: bool
            是否成功落子
        """
        action = pos[0] + pos[1] * self.board_w
        if action in self.available_actions:
            self.do_action(action)
            return True
        return False

    def do_action_print(self, pos: tuple):
        print(f'player {self.current_player}, pos({pos[0]}, {pos[1]})')
        good = self.do_action_with_check(pos)
        if not good:
            print('!!')
            exit(-1)
        else:
            self.print(pos)

    def is_game_over(self) -> Tuple[bool, int]:
        """ 判断游戏是否结束

        Returns
        -------
        is_over: bool
            游戏是否结束，分出胜负则为 `True`, 否则为 `False`

        winner: int
            游戏赢家，有以下几种:
            * 如果游戏分出胜负，则为 `BLACK` 或 `WHITE`，己方可下整张棋盘即为胜
            * 如果没分出胜负，则为 `0`
        """

        if len(self.white_available_points) == self.cell_len:
            return True, self.WHITE
        elif len(self.black_available_points) == self.cell_len:
            return True, self.BLACK
        else:
            return False, 0

    def is_game_over_with_limit(self, max_action=400) -> Tuple[bool, int]:
        is_over, winner = self.is_game_over()
        if self.action_len >= max_action:
            is_over = True
            winner = 0
        return is_over, winner

    def get_state_reward(self, player) -> float:
        white = self.white_count
        black = self.black_count

        # self_factor = 0.5 ** (self.action_len / self.cell_len)  # 前期自己比较重要，后期杀敌比较重要？
        # enemy_factor = 1.0 - self_factor

        self_factor = 0.0
        enemy_factor = 0.0

        # if player == self.WHITE:
        #     return (white * (1 + self_factor) - black * (1 + enemy_factor)) / (white + black)
        # elif player == self.BLACK:
        #     return (black * (1 + self_factor) - white * (1 + enemy_factor)) / (white + black)

        if player == self.WHITE:
            return (white * (1 + self_factor) - black * (1 + enemy_factor)) / 3 / self.cell_len
        elif player == self.BLACK:
            return (black * (1 + self_factor) - white * (1 + enemy_factor)) / 3 / self.cell_len

    def get_feature_planes(self, player) -> torch.Tensor:
        """ 棋盘状态特征张量，维度为 `(n_feature_planes, board_len, board_len)`

        Returns
        -------
        feature_planes: Tensor of shape `(n_feature_planes, board_len, board_len)`
            特征平面图像
        """
        n = self.board_w * self.board_h
        feature_planes = torch.zeros((self.n_feature_planes, n))

        history_len = self.n_feature_planes // 2

        for h in range(history_len):
            if self.n_feature_planes > 2:
                state = self.state_history[h]
            else:
                state = self.state

            for i in range(n):
                cell_state = state.get(i, self.EMPTY)
                if np.sign(cell_state) == player:
                    feature_planes[h * 2, i] = abs(cell_state)
                elif np.sign(cell_state) == -player:
                    feature_planes[h * 2 + 1, i] = -abs(cell_state)

        return feature_planes.view(self.n_feature_planes, self.board_w, self.board_h)

    def print(self, highlight: tuple = None):
        print(' +-', end='')
        for x in range(self.board_w):
            print(f'--{x}--', end='')
        print()
        for y in range(self.board_h):
            print(f'{y}|', end='')
            for x in range(self.board_w):
                index = y * self.board_w + x
                state = self.state.get(index, self.EMPTY)
                if highlight is not None and x == highlight[0] and y == highlight[1]:
                    print(' >', end='')
                else:
                    print('  ', end='')

                if state == 0:
                    print('   ', end='')
                elif state == 1:
                    print(' x ', end='')
                elif state == 2:
                    print('xx ', end='')
                elif state == 3:
                    print('xxx', end='')
                elif state == -1:
                    print(' o ', end='')
                elif state == -2:
                    print('oo ', end='')
                elif state == -3:
                    print('ooo', end='')
            print()

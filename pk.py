# coding:utf-8
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from alphazero import BubbleBoard
from game.agent import Agent
from train import *

pool = ThreadPoolExecutor(max_workers=2)

agents = [
    config_A,
    config_B,
    config_C,
    config_D,
    config_E,
    {
        'model': '0.pth',
        'n_mcts_iters': 500,
        'n_feature_planes': 2,
    }
]
lock = threading.Lock()
futures = []
results = np.zeros([len(agents), len(agents), 3])


# 用两个board来兼容不同feature_plane数量的模型
def pk(board1: BubbleBoard, board2: BubbleBoard, agent1: Agent, agent2: Agent) -> int:
    board1.clear_board()
    board2.clear_board()

    while True:
        action1 = agent1.run()
        board1.do_action(action1)
        board2.do_action(action1)
        is_over, winner = board1.is_game_over_with_limit(200)
        if is_over:
            return winner

        action2 = agent2.run()
        board1.do_action(action2)
        board2.do_action(action2)
        is_over, winner = board1.is_game_over_with_limit(200)
        if is_over:
            return winner


# 两个模型进行pk
def run(i: int, j: int):
    board1 = BubbleBoard(board_len=5, n_feature_planes=agents[i].n_feature_planes)
    board2 = BubbleBoard(board_len=5, n_feature_planes=agents[j].n_feature_planes)
    agent1 = Agent(board1, f'model/{agents[i].model}.pth', n_iters=agents[i].n_mcts_iters)
    agent2 = Agent(board2, f'model/{agents[j].model}.pth', n_iters=agents[j].n_mcts_iters)

    winner = pk(board1, board2, agent1, agent2)
    lock.acquire()
    results[i][j][1 - winner] += 1
    lock.release()
    print(f'finish {i} {j}, winner {winner}')

    winner = pk(board1, board2, agent2, agent1)
    lock.acquire()
    results[i][j][1 + winner] += 1
    lock.release()
    print(f'finish {j} {i}, winner {winner}')


def main():
    n = 25

    # 单循环
    # for i in range(len(agents) - 1):
    #     for j in range(i + 1, len(agents)):
    #         for _ in range(n):
    #             futures.append(pool.submit(run, i, j))

    # 测试相邻的网络
    # for i in range(len(agents) - 1):
    #     for _ in range(n):
    #         futures.append(pool.submit(run, i, i + 1))

    for _ in range(n):
        futures.append(pool.submit(run, 0, 1))
    for _ in range(n):
        futures.append(pool.submit(run, 0, 2))
    for _ in range(n):
        futures.append(pool.submit(run, 5, 2))
    for _ in range(n):
        futures.append(pool.submit(run, 2, 3))
    for _ in range(n):
        futures.append(pool.submit(run, 3, 4))

    for f in futures:
        f.result()

    print()
    print('results:')
    for i in range(len(agents) - 1):
        for j in range(i + 1, len(agents)):
            r = results[i][j]
            print(f'{agents[i].model} {int(r[0])} : {int(r[1])} : {int(r[2])} {agents[j].model}')


if __name__ == '__main__':
    main()

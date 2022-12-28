# coding:utf-8
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from alphazero import BubbleBoard
from game.agent import Agent

pool = ThreadPoolExecutor(max_workers=2)

agents = [
    "checkpoint/saved_bubble_reward_0",
    "checkpoint/saved_bubble_reward_50",
    "checkpoint/saved_bubble_reward_100",
    "checkpoint/saved_bubble_reward_150",
    "checkpoint/saved_bubble_reward_200",
]
lock = threading.Lock()
futures = []
results = np.zeros([len(agents), len(agents), 3])


def pk(board: BubbleBoard, agent1: Agent, agent2: Agent) -> int:
    board.clear_board()

    while True:
        action1 = agent1.run()
        board.do_action(action1)
        is_over, winner = board.is_game_over_with_limit(200)
        if is_over:
            return winner

        action2 = agent2.run()
        board.do_action(action2)
        is_over, winner = board.is_game_over_with_limit(200)
        if is_over:
            return winner


def run(i: int, j: int):
    board = BubbleBoard(board_len=5)
    agent1 = Agent(board, f'model/{agents[i]}.pth')
    agent2 = Agent(board, f'model/{agents[j]}.pth')

    # print(f'begin {i} {j}')
    winner = pk(board, agent1, agent2)
    lock.acquire()
    results[i][j][1 - winner] += 1
    lock.release()
    print(f'finish {i} {j}, winner {winner}')

    # print(f'begin {j} {i}')
    winner = pk(board, agent2, agent1)
    lock.acquire()
    results[i][j][1 + winner] += 1
    lock.release()
    print(f'finish {j} {i}, winner {winner}')


def main():
    n = 10

    # for i in range(len(agents) - 1):
    #     for j in range(i + 1, len(agents)):
    #         for _ in range(n):
    #             futures.append(pool.submit(run, i, j))

    for i in range(len(agents) - 1):
        for _ in range(n):
            futures.append(pool.submit(run, i, i + 1))

    for f in futures:
        f.result()

    print()
    print('results:')
    for i in range(len(agents) - 1):
        for j in range(i + 1, len(agents)):
            r = results[i][j]
            print(f'{agents[i]} {int(r[0])} : {int(r[1])} : {int(r[2])} {agents[j]}')


if __name__ == '__main__':
    main()

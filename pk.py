# coding:utf-8
import threading

from alphazero import BubbleBoard
from game.agent import Agent


def pk(board: BubbleBoard, agent1: Agent, agent2: Agent) -> int:
    print("pk")
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


def pk_n(n: int, board: BubbleBoard, agent1: Agent, agent2: Agent) -> [int, int]:
    w1_count = 0
    w2_count = 0

    for _ in range(n):
        w1 = pk(board, agent1, agent2)
        if w1 > 0:
            w1_count += 1
        elif w1 < 0:
            w2_count += 1

        w2 = pk(board, agent2, agent1)
        if w2 > 0:
            w2_count += 1
        elif w2 < 0:
            w1_count += 1

    return [w1_count, w2_count]


agent1_win = 0
agent2_win = 0
lock = threading.Lock()


def run(n: int, name1: str, name2: str):
    board = BubbleBoard(board_len=5)
    agent1 = Agent(board, f'model/history/{name1}.pth')
    agent2 = Agent(board, f'model/history/{name2}.pth')
    [w1, w2] = pk_n(n, board, agent1, agent2)
    print(f'agent1 {w1} : {w2} agent2')
    global agent1_win, agent2_win
    lock.acquire()
    agent1_win += w1
    agent2_win += w2
    lock.release()


if __name__ == '__main__':

    # agent1_name = "uninitialized"
    # agent2_name = "bubble_reward_100"

    # agent1_name = "cell_reward_100"
    # agent2_name = "uninitialized"

    agent1_name = "cell_reward_100"
    agent2_name = "bubble_reward_100"

    threads = []
    thread_count = 2
    pairs_per_thread = 25

    for _ in range(thread_count):
        thread = threading.Thread(target=run, args=(pairs_per_thread, agent1_name, agent2_name))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    print()
    total = thread_count * pairs_per_thread * 2
    draw = total - agent1_win - agent2_win
    print(f'total: {total}')
    print(f'{agent1_name} {agent1_win} : {draw} : {agent2_win} {agent2_name}')

# coding:utf-8

from alphazero import BubbleBoard
from game.agent import Agent

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

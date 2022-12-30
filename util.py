from alphazero.bubble_board import BubbleBoard

board = BubbleBoard(board_len=5, n_feature_planes=4)


p0 = board.get_feature_planes(1)
board.do_action_print((3, 3))
p1 = board.get_feature_planes(1)
board.do_action_print((4, 4))
p2 = board.get_feature_planes(1)

board.do_action_print((3, 3))
board.do_action_print((4, 4))
board.do_action_print((3, 3))
board.do_action_print((4, 4))
board.do_action_print((3, 3))
print(f'{board.get_state_reward(1)}')
print(f'{board.get_state_reward(-1)}')
is_over, winner = board.is_game_over()
print(f'{is_over}, {winner}')

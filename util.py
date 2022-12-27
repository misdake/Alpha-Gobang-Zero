from alphazero.bubble_board import BubbleBoard

board = BubbleBoard(board_len=5)


board.do_action_print((3, 3))
board.do_action_print((4, 4))
board.do_action_print((3, 3))
board.do_action_print((4, 4))
board.do_action_print((3, 3))
board.do_action_print((4, 4))
board.do_action_print((3, 3))
is_over, winner = board.is_game_over()
print(f'{is_over}, {winner}')

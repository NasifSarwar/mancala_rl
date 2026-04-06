from game.mancala_board import MancalaBoard

board = MancalaBoard()
board.render()

# Manually play a few moves
while not board.done:
    player = board.current_player
    legal = board.get_legal_moves(player)
    print(f"Player {player}, choose a pit from {legal}: ", end="")
    pit = int(input())
    if pit not in legal:
        print("Illegal move! Try again.")
        continue
    extra, captured = board.make_move(player, pit)
    board.render()
    if extra:
        print(">> Extra turn!")
    if captured:
        print(f">> Captured {captured} stones!")
    if not extra:
        board.current_player = 1 - player

print(f"\nGame over! Winner: Player {board.winner}")
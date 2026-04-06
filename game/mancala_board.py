import numpy as np

class MancalaBoard:
    """
    Represents the Mancala (Kalah) board as a 14-element array.

    Layout:
      Indices 0-5   -> Player 0's pits
      Index   6     -> Player 0's store
      Indices 7-12  -> Player 1's pits
      Index   13    -> Player 1's store

    Starting state: 4 stones in every pit, stores empty.
    """

    # Which index is each player's store
    STORE = {0: 6, 1: 13}

    # Player 0's pits: 0-5,  Player 1's pits: 7-12
    PITS = {0: list(range(0, 6)), 1: list(range(7, 13))}

    def __init__(self):
        self.reset()

    def reset(self):
        """Set board back to the starting state."""
        self.board = np.array([4]*6 + [0] + [4]*6 + [0], dtype=np.int32)
        self.current_player = 0   # Player 0 always goes first
        self.done = False
        self.winner = None        # 0, 1, or -1 (draw)

    # ------------------------------------------------------------------
    # Helper: pit index on the board for a given player + pit number 0-5
    # ------------------------------------------------------------------
    def _board_index(self, player, pit):
        """Convert local pit number (0-5) to board array index."""
        return self.PITS[player][pit]

    def get_legal_moves(self, player):
        """Return list of local pit numbers (0-5) that are not empty."""
        return [i for i in range(6) if self.board[self._board_index(player, i)] > 0]

    # ------------------------------------------------------------------
    # Core: make a move
    # ------------------------------------------------------------------
    def make_move(self, player, pit):
        """
        Sow stones from pit (0-5) for the given player.

        Returns:
            extra_turn (bool) - True if the player earned another turn
            captured   (int)  - number of stones captured this move
        """
        board_idx = self._board_index(player, pit)

        # Pick up all stones from the chosen pit
        stones = self.board[board_idx]
        self.board[board_idx] = 0

        opponent_store = self.STORE[1 - player]
        extra_turn = False
        captured = 0

        # Sow counter-clockwise (just increment index, wrapping at 14)
        idx = board_idx
        while stones > 0:
            idx = (idx + 1) % 14

            # Skip the opponent's store
            if idx == opponent_store:
                continue

            self.board[idx] += 1
            stones -= 1

        # --- Rule: Extra turn ---
        if idx == self.STORE[player]:
            extra_turn = True

        # --- Rule: Capture ---
        # Last stone lands on an empty pit on YOUR side (and it was empty before)
        player_pits = self.PITS[player]
        if idx in player_pits and self.board[idx] == 1:
            opposite_idx = 12 - idx   # mirror index across the board
            if self.board[opposite_idx] > 0:
                captured = self.board[opposite_idx] + 1   # opposite + the stone we just placed
                self.board[self.STORE[player]] += captured
                self.board[idx] = 0
                self.board[opposite_idx] = 0

        # --- Check terminal state ---
        self._check_game_over()

        return extra_turn, captured

    # ------------------------------------------------------------------
    # Terminal state check
    # ------------------------------------------------------------------
    def _check_game_over(self):
        """
        Game ends when ALL pits on either side are empty.
        Remaining stones on the other side go to that player's store.
        """
        p0_empty = all(self.board[i] == 0 for i in self.PITS[0])
        p1_empty = all(self.board[i] == 0 for i in self.PITS[1])

        if p0_empty or p1_empty:
            # Sweep remaining stones
            for i in self.PITS[0]:
                self.board[self.STORE[0]] += self.board[i]
                self.board[i] = 0
            for i in self.PITS[1]:
                self.board[self.STORE[1]] += self.board[i]
                self.board[i] = 0

            self.done = True

            s0 = self.board[self.STORE[0]]
            s1 = self.board[self.STORE[1]]

            if s0 > s1:
                self.winner = 0
            elif s1 > s0:
                self.winner = 1
            else:
                self.winner = -1  # draw

    # ------------------------------------------------------------------
    # Pretty print for terminal testing
    # ------------------------------------------------------------------
    def render(self):
        b = self.board.tolist()  # converts np.int32 to plain int
        print("\n  --- Mancala Board ---")
        print(f"  P1 store : {b[13]}")
        print(f"  P1 pits  : {b[7:13][::-1]}  (right to left: 12..7)")
        print(f"  P0 pits  : {b[0:6]}  (left to right: 0..5)")
        print(f"  P0 store : {b[6]}")
        print(f"  Current player: {self.current_player}")
        print(f"  Legal moves   : {self.get_legal_moves(self.current_player)}")
        print()
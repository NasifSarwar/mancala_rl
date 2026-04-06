import random


class RandomAgent:
    """
    Picks a random legal move every turn.
    This is our baseline opponent — the AI should learn to beat this easily.
    """

    def __init__(self, player):
        self.player = player   # 0 or 1

    def select_action(self, legal_moves):
        return random.choice(legal_moves)
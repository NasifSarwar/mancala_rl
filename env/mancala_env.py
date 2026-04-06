import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector

from game.mancala_board import MancalaBoard


class MancalaEnv(AECEnv):
    """
    PettingZoo AEC (Agent-Environment-Cycle) wrapper for Mancala.

    Two agents take turns. Each sees the full 14-element board.
    Actions are 0-5 (which pit to sow from).
    """

    metadata = {"name": "mancala_v0"}

    def __init__(self):
        super().__init__()

        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]

        # Each agent sees the full board: 14 integers, each 0-48
        self.observation_spaces = {
            a: spaces.Box(low=0, high=48, shape=(14,), dtype=np.int32)
            for a in self.agents
        }

        # Each agent picks one of 6 pits
        self.action_spaces = {
            a: spaces.Discrete(6)
            for a in self.agents
        }

        self.board = MancalaBoard()
        self._agent_selector = agent_selector(self.agents)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        self.board.reset()
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards        = {a: 0 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations   = {a: False for a in self.agents}
        self.truncations    = {a: False for a in self.agents}
        self.infos          = {a: {} for a in self.agents}

    # ------------------------------------------------------------------
    def observe(self, agent):
        """Return the board state as a numpy array."""
        return self.board.board.copy()

    # ------------------------------------------------------------------
    def step(self, action):
        agent = self.agent_selection
        player = int(agent[-1])   # "player_0" -> 0, "player_1" -> 1

        # Clear previous rewards
        self.rewards = {a: 0 for a in self.agents}

        # --- Illegal move penalty ---
        legal = self.board.get_legal_moves(player)
        if action not in legal:
            self.rewards[agent] = -10
            # End the game on illegal move
            self.terminations = {a: True for a in self.agents}
            self.agent_selection = self._agent_selector.next()
            self._accumulate_rewards()
            return

        # --- Make the move ---
        extra_turn, captured = self.board.make_move(player, action)

        # --- Reward shaping ---
        reward = 0
        if captured > 0:
            reward += 0.1 * captured   # small bonus for capturing
        if extra_turn:
            reward += 0.1              # small bonus for extra turn

        # --- Terminal rewards ---
        if self.board.done:
            self.terminations = {a: True for a in self.agents}
            if self.board.winner == -1:
                # Draw
                self.rewards["player_0"] = 0
                self.rewards["player_1"] = 0
            else:
                winner_agent = f"player_{self.board.winner}"
                loser_agent  = f"player_{1 - self.board.winner}"
                self.rewards[winner_agent] = 1
                self.rewards[loser_agent]  = -1
        else:
            self.rewards[agent] += reward

        # --- Switch turn ---
        if extra_turn and not self.board.done:
            # Same player goes again — don't advance the selector
            pass
        else:
            self.agent_selection = self._agent_selector.next()

        self._accumulate_rewards()

    # ------------------------------------------------------------------
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # ------------------------------------------------------------------
    def render(self):
        self.board.render()

    def close(self):
        pass
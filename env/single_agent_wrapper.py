import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.mancala_env import MancalaEnv
from agents.random_agent import RandomAgent


class SingleAgentWrapper(gym.Env):
    """
    Wraps MancalaEnv so Stable-Baselines3 can train on it.

    SB3 only understands single-agent environments.
    This wrapper makes the AI always play as player_0.
    Player_1 is handled internally by an opponent (RandomAgent or a loaded model).
    """

    def __init__(self, opponent=None):
        super().__init__()

        self.env = MancalaEnv()

        # If no opponent given, use a random agent
        self.opponent = opponent if opponent is not None else RandomAgent(1)

        # SB3 needs these two defined
        self.observation_space = spaces.Box(low=0, high=48, shape=(14,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

    def reset(self, seed=None, options=None):
        self.env.reset()
        obs = self.env.observe("player_0").astype(np.float32)
        return obs, {}

    def step(self, action):
        # --- Player 0 (our AI) takes its action ---
        legal = self.env.board.get_legal_moves(0)

        # If illegal, pick a random legal move instead (avoids -10 penalty during early training)
        if action not in legal:
            action = np.random.choice(legal) if legal else 0

        self.env.step(action)

        # --- Let Player 1 (opponent) play until it's Player 0's turn again ---
        while (
            not all(self.env.terminations.values())
            and self.env.agent_selection == "player_1"
        ):
            legal_opp = self.env.board.get_legal_moves(1)
            if not legal_opp:
                break

            # Opponent can be RandomAgent or a trained SB3 model
            if hasattr(self.opponent, "predict"):
                # SB3 model
                opp_obs = self.env.observe("player_1").astype(np.float32)
                opp_action, _ = self.opponent.predict(opp_obs, deterministic=True)
                opp_action = int(opp_action)
                if opp_action not in legal_opp:
                    opp_action = np.random.choice(legal_opp)
            else:
                # RandomAgent
                opp_action = self.opponent.select_action(legal_opp)

            self.env.step(opp_action)

        # --- Collect results ---
        done = all(self.env.terminations.values())
        reward = self.env._cumulative_rewards.get("player_0", 0)
        obs = self.env.observe("player_0").astype(np.float32)

        return obs, reward, done, False, {}

    def render(self):
        self.env.render()
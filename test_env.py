from env.mancala_env import MancalaEnv
import random

env = MancalaEnv()
env.reset()

for step in range(50):
    agent = env.agent_selection
    player = int(agent[-1])
    legal = env.board.get_legal_moves(player)
    action = random.choice(legal)   # pick random legal move
    env.step(action)
    if all(env.terminations.values()):
        print(f"Game over at step {step}!")
        print(f"Winner: player_{env.board.winner}")
        break

print("Environment works correctly.")
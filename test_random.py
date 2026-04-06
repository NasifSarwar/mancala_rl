from env.mancala_env import MancalaEnv
from agents.random_agent import RandomAgent

env = MancalaEnv()
agents = {
    "player_0": RandomAgent(0),
    "player_1": RandomAgent(1),
}

wins = {0: 0, 1: 0, "draw": 0}

for game in range(1000):
    env.reset()
    while not all(env.terminations.values()):
        agent_name = env.agent_selection
        player = int(agent_name[-1])
        legal = env.board.get_legal_moves(player)
        action = agents[agent_name].select_action(legal)
        env.step(action)

    winner = env.board.winner
    if winner == -1:
        wins["draw"] += 1
    else:
        wins[winner] += 1

print(f"Results over 1000 games:")
print(f"  Player 0 wins : {wins[0]}")
print(f"  Player 1 wins : {wins[1]}")
print(f"  Draws         : {wins['draw']}")
print("\nBaseline done. Environment is stable.")
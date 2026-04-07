from stable_baselines3 import PPO
from env.single_agent_wrapper import SingleAgentWrapper
from agents.random_agent import RandomAgent

def evaluate(model, n_games=200):
    env = SingleAgentWrapper(opponent=RandomAgent(1))
    wins, losses, draws = 0, 0, 0

    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(int(action))

        winner = env.env.board.winner
        if winner == 0:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            draws += 1

    print(f"\nResults over {n_games} games vs Random Agent:")
    print(f"  Wins   : {wins}  ({100*wins/n_games:.1f}%)")
    print(f"  Losses : {losses}  ({100*losses/n_games:.1f}%)")
    print(f"  Draws  : {draws}  ({100*draws/n_games:.1f}%)")

print("--- Stage 1 model (trained vs Random) ---")
stage1 = PPO.load("models/stage1_vs_random")
evaluate(stage1)

print("\n--- Stage 2 model (Self-play) ---")
stage2 = PPO.load("models/stage2_selfplay")
evaluate(stage2)
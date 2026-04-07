import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env.single_agent_wrapper import SingleAgentWrapper
from agents.random_agent import RandomAgent

# --- Create folders ---
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ---------------------------------------------------------------
# STAGE 1: Train against Random Agent
# ---------------------------------------------------------------
print("Stage 1: Training against Random Agent...")

env = SingleAgentWrapper(opponent=RandomAgent(1))
check_env(env, warn=True)  # Verify the env is set up correctly

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
)

model.learn(total_timesteps=200_000, tb_log_name="stage1_vs_random")
model.save("models/stage1_vs_random")
print("Stage 1 done. Model saved to models/stage1_vs_random.zip")

# ---------------------------------------------------------------
# STAGE 2: Self-Play — train against the Stage 1 model
# ---------------------------------------------------------------
print("\nStage 2: Self-play against Stage 1 model...")

# Load stage 1 model as the opponent
stage1_opponent = PPO.load("models/stage1_vs_random")

env2 = SingleAgentWrapper(opponent=stage1_opponent)

model2 = PPO(
    "MlpPolicy",
    env2,
    verbose=1,
    tensorboard_log="./logs/",
    learning_rate=1e-4,   # slower learning rate for fine-tuning
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
)

model2.learn(total_timesteps=300_000, tb_log_name="stage2_selfplay")
model2.save("models/stage2_selfplay")
print("Stage 2 done. Final model saved to models/stage2_selfplay.zip")
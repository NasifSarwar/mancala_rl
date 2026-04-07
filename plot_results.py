"""
plot_results.py

Generates 3 clean plots for the presentation:
  1. Win rate: Stage 1 vs Random over simulated games
  2. Win rate: Stage 2 vs Random over simulated games
  3. Head-to-head: Stage 2 vs Stage 1 win rate over games
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stable_baselines3 import PPO
from env.single_agent_wrapper import SingleAgentWrapper
from agents.random_agent import RandomAgent
from game.mancala_board import MancalaBoard
import os

os.makedirs("plots", exist_ok=True)

# -------------------------------------------------------------------
# Style
# -------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#16161e",
    "axes.facecolor":   "#1e1e2e",
    "axes.edgecolor":   "#444466",
    "axes.labelcolor":  "#ccccdd",
    "xtick.color":      "#888899",
    "ytick.color":      "#888899",
    "text.color":       "#ccccdd",
    "grid.color":       "#2a2a3e",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.7,
    "font.family":      "DejaVu Sans",
})

BLUE   = "#64aaff"
ORANGE = "#ff9050"
GREEN  = "#50e090"
RED    = "#ff6070"
PURPLE = "#c078f0"
WHITE  = "#f0f0f5"

# -------------------------------------------------------------------
# Helper: evaluate model over N games, return rolling win rate
# -------------------------------------------------------------------
def rolling_winrate(model, n_games=300, window=30, opponent=None):
    env = SingleAgentWrapper(opponent=opponent or RandomAgent(1))
    results = []
    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(int(action))
        results.append(1 if env.env.board.winner == 0 else 0)

    # Rolling average
    rates = []
    for i in range(len(results)):
        chunk = results[max(0, i - window):i + 1]
        rates.append(sum(chunk) / len(chunk))
    return rates

# -------------------------------------------------------------------
# Helper: head to head between two models
# -------------------------------------------------------------------
def head_to_head(model_a, model_b, n_games=300):
    a_wins, b_wins, draw_list = [], [], []
    a_total = b_total = d_total = 0

    for _ in range(n_games):
        board = MancalaBoard()
        current = 0
        while not board.done:
            legal = board.get_legal_moves(current)
            if not legal:
                current = 1 - current
                continue
            obs = board.board.astype(np.float32)
            if current == 0:
                action, _ = model_a.predict(obs, deterministic=True)
            else:
                flipped = np.concatenate([obs[7:14], obs[0:7]])
                action, _ = model_b.predict(flipped, deterministic=True)
            action = int(action)
            if action not in legal:
                action = np.random.choice(legal)
            extra, _ = board.make_move(current, action)
            if not extra:
                current = 1 - current

        if board.winner == 0:   a_total += 1
        elif board.winner == 1: b_total += 1
        else:                   d_total += 1

        total = a_total + b_total + d_total
        a_wins.append(a_total / total)
        b_wins.append(b_total / total)
        draw_list.append(d_total / total)

    return a_wins, b_wins, draw_list

# -------------------------------------------------------------------
print("Evaluating Stage 1 vs Random...")
s1 = PPO.load("models/stage1_vs_random")
s1_rates = rolling_winrate(s1, n_games=300)

print("Evaluating Stage 2 vs Random...")
s2 = PPO.load("models/stage2_selfplay")
s2_rates = rolling_winrate(s2, n_games=300)

print("Head-to-head Stage 2 vs Stage 1...")
h2h_s2, h2h_s1, h2h_draw = head_to_head(s2, s1, n_games=300)

# -------------------------------------------------------------------
# Plot 1: Stage 1 win rate vs Random
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(s1_rates))
ax.plot(x, s1_rates, color=BLUE, linewidth=2, label="Stage 1 win rate")
ax.axhline(0.5, color="#555577", linestyle="--", linewidth=1, label="50% baseline")
ax.fill_between(x, s1_rates, 0.5, where=[v > 0.5 for v in s1_rates],
                alpha=0.15, color=BLUE)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Game number", fontsize=12)
ax.set_ylabel("Win rate (rolling avg)", fontsize=12)
ax.set_title("Stage 1 — Win Rate vs Random Agent", fontsize=14, color=BLUE, pad=12)
ax.legend(fontsize=11)
ax.grid(True)
plt.tight_layout()
plt.savefig("plots/stage1_winrate.png", dpi=150, facecolor=fig.get_facecolor())
print("Saved plots/stage1_winrate.png")
plt.close()

# -------------------------------------------------------------------
# Plot 2: Stage 2 win rate vs Random
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(s2_rates))
ax.plot(x, s2_rates, color=ORANGE, linewidth=2, label="Stage 2 win rate")
ax.axhline(0.5, color="#555577", linestyle="--", linewidth=1, label="50% baseline")
ax.fill_between(x, s2_rates, 0.5, where=[v > 0.5 for v in s2_rates],
                alpha=0.15, color=ORANGE)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Game number", fontsize=12)
ax.set_ylabel("Win rate (rolling avg)", fontsize=12)
ax.set_title("Stage 2 — Win Rate vs Random Agent", fontsize=14, color=ORANGE, pad=12)
ax.legend(fontsize=11)
ax.grid(True)
plt.tight_layout()
plt.savefig("plots/stage2_winrate.png", dpi=150, facecolor=fig.get_facecolor())
print("Saved plots/stage2_winrate.png")
plt.close()

# -------------------------------------------------------------------
# Plot 3: Head to head Stage 2 vs Stage 1
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(h2h_s2))
ax.plot(x, h2h_s2,   color=ORANGE, linewidth=2, label="Stage 2 wins")
ax.plot(x, h2h_s1,   color=BLUE,   linewidth=2, label="Stage 1 wins")
ax.plot(x, h2h_draw, color=GREEN,  linewidth=1.5, linestyle="--", label="Draws")
ax.axhline(0.5, color="#555577", linestyle=":", linewidth=1)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Game number", fontsize=12)
ax.set_ylabel("Cumulative win rate", fontsize=12)
ax.set_title("Head-to-Head: Stage 2 vs Stage 1", fontsize=14, color=PURPLE, pad=12)
ax.legend(fontsize=11)
ax.grid(True)
plt.tight_layout()
plt.savefig("plots/head_to_head.png", dpi=150, facecolor=fig.get_facecolor())
print("Saved plots/head_to_head.png")
plt.close()

# -------------------------------------------------------------------
# Plot 4: Summary bar chart
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
categories = ["Stage 1\nvs Random", "Stage 2\nvs Random", "Stage 2\nvs Stage 1"]
win_rates  = [s1_rates[-1]*100, s2_rates[-1]*100, h2h_s2[-1]*100]
colors     = [BLUE, ORANGE, PURPLE]

bars = ax.bar(categories, win_rates, color=colors, width=0.5, edgecolor="#333355")
ax.axhline(50, color="#555577", linestyle="--", linewidth=1, label="50% baseline")

for bar, val in zip(bars, win_rates):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1.5,
            f"{val:.1f}%", ha="center", va="bottom",
            fontsize=13, color=WHITE if val > 0 else "#888899",
            fontweight="bold")

ax.set_ylim(0, 110)
ax.set_ylabel("Win rate %", fontsize=12)
ax.set_title("Performance Summary", fontsize=14, color=PURPLE, pad=12)
ax.legend(fontsize=11)
ax.grid(True, axis="y")
plt.tight_layout()
plt.savefig("plots/summary.png", dpi=150, facecolor=fig.get_facecolor())
print("Saved plots/summary.png")
plt.close()

print("\nAll plots saved to plots/ folder.")
WHITE = "#f0f0f5"
# Mancala RL — Reinforcement Learning for Mancala (Kalah)

Lab 3 · Machine Learning 2026 · Åbo Akademi University

An AI agent that learns to play Mancala from scratch using Reinforcement Learning (PPO).  
No hand-coded rules — the agent learns purely through trial and error.

---

## Quick Start

### 1. Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### 2. Clone and set up

```bash
git clone <your-repo-url>
cd mancala_rl

# Create virtual environment
uv venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
uv pip install numpy pettingzoo stable-baselines3 gymnasium pygame matplotlib tensorboard
```

### 3. Train the models

```bash
python train.py
```

This runs two training stages and saves models to `models/`.  
Takes about 3-5 minutes on a modern laptop.

### 4. Play against the AI

```bash
python play.py
```

Click a green pit to make your move. Press **R** to restart.

---

## All Scripts

| Script | What it does | Command |
|---|---|---|
| `train.py` | Train Stage 1 (vs random) and Stage 2 (self-play) | `python train.py` |
| `evaluate.py` | Print win/loss/draw rates for both models | `python evaluate.py` |
| `play.py` | Play against the trained AI with Pygame UI | `python play.py` |
| `watch_training.py` | Watch AI vs AI with live win-rate graph | `python watch_training.py` |
| `plot_results.py` | Generate 4 graphs saved to `plots/` folder | `python plot_results.py` |
| `test_game.py` | Manually play in the terminal (testing only) | `python test_game.py` |
| `test_env.py` | Verify the PettingZoo environment works | `python test_env.py` |
| `test_random.py` | Run 1000 random vs random games | `python test_random.py` |

---

## Project Structure

```
mancala_rl/
├── game/
│   └── mancala_board.py        # Core game engine — rules, sowing, captures
├── env/
│   ├── mancala_env.py          # PettingZoo AEC environment wrapper
│   └── single_agent_wrapper.py # Gymnasium wrapper for SB3 compatibility
├── agents/
│   └── random_agent.py         # Baseline random opponent
├── models/                     # Saved trained models (created after training)
│   ├── stage1_vs_random.zip
│   └── stage2_selfplay.zip
├── plots/                      # Generated graphs (created after plot_results.py)
├── logs/                       # TensorBoard logs (created during training)
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation script
├── play.py                     # Human vs AI game
├── watch_training.py           # AI vs AI viewer with live graph
└── plot_results.py             # Generate plots for slides
```

---

## How it Works

### The Board
The board is a 14-element array:
```
[P0_pit0, P0_pit1, P0_pit2, P0_pit3, P0_pit4, P0_pit5, P0_store,
 P1_pit0, P1_pit1, P1_pit2, P1_pit3, P1_pit4, P1_pit5, P1_store]
```
Starting state: `[4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]`

### Training Curriculum
1. **Stage 1** — 200,000 steps vs RandomAgent → agent learns basics, wins ~91.5%
2. **Stage 2** — 300,000 steps vs Stage 1 model (self-play) → agent learns deeper strategy

### Algorithm
PPO (Proximal Policy Optimization) from Stable-Baselines3.  
Chosen for stability with discrete action spaces and compatibility with curriculum learning.

### Results
| Matchup | Win Rate |
|---|---|
| Stage 1 vs Random | 91.5% |
| Stage 2 vs Random | 74.5% |
| Stage 2 vs Stage 1 | ~55-60% |

---

## Controls

**play.py**
- Click a green pit → make your move
- `R` → restart the game

**watch_training.py**
- `↑` arrow → speed up
- `↓` arrow → slow down

---

## Notes

- Models are saved as `.zip` files in `models/` — do not delete these before running `play.py`
- `plots/` and `models/` are in `.gitignore` — teammates need to run `train.py` themselves
- Training is fast (~4 min total) so re-training locally is recommended

"""
watch_training.py

Shows AI vs AI games live on the left panel.
On the right panel, a live win-rate graph updates after every game.
Uses the already-trained Stage 1 and Stage 2 models playing against each other.
"""

import pygame
import sys
import numpy as np
from collections import deque
from stable_baselines3 import PPO
from game.mancala_board import MancalaBoard

# -------------------------------------------------------------------
# Colours
# -------------------------------------------------------------------
BG         = (18,  18,  28)
BOARD_BG   = (36,  36,  56)
PIT_0      = (80,  160, 120)   # Stage 2 (blue agent)
PIT_1      = (180,  90, 120)   # Stage 1 (red agent)
PIT_EMPTY  = (50,   50,  75)
STORE_BG   = (28,  28,  44)
TEXT       = (220, 220, 235)
PANEL_BG   = (24,  24,  38)
GRID_COL   = (45,  45,  65)
LINE_S1    = (100, 170, 255)   # stage1 win rate line
LINE_S2    = (255, 140,  80)   # stage2 win rate line
ACCENT     = (160, 110, 255)
WHITE      = (240, 240, 245)

# -------------------------------------------------------------------
# Window: left = board, right = graph panel
# -------------------------------------------------------------------
W, H       = 1180, 520
BOARD_W    = 680
PANEL_X    = BOARD_W
PANEL_W    = W - BOARD_W

PIT_R      = 38
STORE_W_   = 62
STORE_H_   = 190
ROW_Y      = [H // 3, int(H * 0.67)]
PIT_XS     = [120 + i * 84 for i in range(6)]
SL         = (50,  H // 2)
SR         = (630, H // 2)

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Mancala — AI vs AI  |  Model Comparison")
clock  = pygame.time.Clock()

font_title = pygame.font.SysFont("Georgia", 20, bold=True)
font_med   = pygame.font.SysFont("Georgia", 17)
font_small = pygame.font.SysFont("Georgia", 13)
font_tiny  = pygame.font.SysFont("Georgia", 11)

# -------------------------------------------------------------------
# Load models
# -------------------------------------------------------------------
model_s2 = PPO.load("models/stage2_selfplay")   # player 0
model_s1 = PPO.load("models/stage1_vs_random")  # player 1

# -------------------------------------------------------------------
# Stats tracking
# -------------------------------------------------------------------
MAX_HISTORY  = 100
win_history  = deque(maxlen=MAX_HISTORY)   # 0 = s2 wins, 1 = s1 wins, -1 = draw
game_count   = 0
s2_wins = s1_wins = draws = 0

# Smoothed win rate over last N games
WINDOW = 20

def smooth_winrate(history, player):
    recent = list(history)[-WINDOW:]
    if not recent:
        return 0.5
    return sum(1 for r in recent if r == player) / len(recent)

# -------------------------------------------------------------------
# Board drawing helpers
# -------------------------------------------------------------------
def draw_pit_w(surface, cx, cy, stones, player, radius=PIT_R):
    col = (PIT_0 if player == 0 else PIT_1) if stones > 0 else PIT_EMPTY
    pygame.draw.circle(surface, col, (cx, cy), radius)
    pygame.draw.circle(surface, WHITE, (cx, cy), radius, 2)
    t = font_med.render(str(stones), True, WHITE)
    surface.blit(t, t.get_rect(center=(cx, cy)))

def draw_store_w(surface, cx, cy, stones, label, col):
    r = pygame.Rect(cx - STORE_W_//2, cy - STORE_H_//2, STORE_W_, STORE_H_)
    pygame.draw.rect(surface, STORE_BG, r, border_radius=14)
    pygame.draw.rect(surface, col, r, 2, border_radius=14)
    lbl = font_small.render(label, True, col)
    surface.blit(lbl, lbl.get_rect(center=(cx, cy - STORE_H_//2 + 16)))
    t = font_med.render(str(stones), True, WHITE)
    surface.blit(t, t.get_rect(center=(cx, cy + 8)))

def draw_game_board(b, current_player):
    # Board card
    card = pygame.Rect(18, H//2 - 160, BOARD_W - 36, 320)
    pygame.draw.rect(screen, BOARD_BG, card, border_radius=22)

    # Row labels
    s2_lbl = font_small.render("Stage 2", True, PIT_0)
    s1_lbl = font_small.render("Stage 1", True, PIT_1)
    screen.blit(s2_lbl, s2_lbl.get_rect(center=(card.left + 40, ROW_Y[1])))
    screen.blit(s1_lbl, s1_lbl.get_rect(center=(card.left + 40, ROW_Y[0])))

    # Stage 1 pits — top (indices 12..7)
    for i in range(6):
        draw_pit_w(screen, PIT_XS[i], ROW_Y[0], b[12 - i], 1)

    # Stage 2 pits — bottom (indices 0..5)
    for i in range(6):
        draw_pit_w(screen, PIT_XS[i], ROW_Y[1], b[i], 0)

    draw_store_w(screen, SL[0], SL[1], b[6],  "S2", PIT_0)
    draw_store_w(screen, SR[0], SR[1], b[13], "S1", PIT_1)

    # Turn indicator
    who = "Stage 2's turn" if current_player == 0 else "Stage 1's turn"
    col = PIT_0 if current_player == 0 else PIT_1
    ind = font_title.render(who, True, col)
    screen.blit(ind, ind.get_rect(center=(BOARD_W // 2, 26)))

    # Score
    score = f"S2: {b[6]}   :   {b[13]} :S1"
    sc = font_med.render(score, True, TEXT)
    screen.blit(sc, sc.get_rect(center=(BOARD_W // 2, H - 20)))

# -------------------------------------------------------------------
# Graph drawing
# -------------------------------------------------------------------
GRAPH_MARGIN = 40
GRAPH_X  = PANEL_X + GRAPH_MARGIN
GRAPH_Y  = 80
GRAPH_W  = PANEL_W - GRAPH_MARGIN * 2
GRAPH_H  = H - GRAPH_Y - 80

def draw_graph():
    # Panel background
    pygame.draw.rect(screen, PANEL_BG,
                     pygame.Rect(PANEL_X, 0, PANEL_W, H))
    pygame.draw.line(screen, GRID_COL, (PANEL_X, 0), (PANEL_X, H), 2)

    # Title
    t = font_title.render("Model Comparison", True, ACCENT)
    screen.blit(t, t.get_rect(center=(PANEL_X + PANEL_W//2, 28)))

    # Stats row
    total = max(game_count, 1)
    stats = f"Games: {game_count}   S2: {s2_wins}  S1: {s1_wins}  D: {draws}"
    st = font_small.render(stats, True, TEXT)
    screen.blit(st, st.get_rect(center=(PANEL_X + PANEL_W//2, 52)))

    # Graph area
    gx, gy, gw, gh = GRAPH_X, GRAPH_Y, GRAPH_W, GRAPH_H
    pygame.draw.rect(screen, (30, 30, 48), pygame.Rect(gx, gy, gw, gh))
    pygame.draw.rect(screen, GRID_COL, pygame.Rect(gx, gy, gw, gh), 1)

    # Horizontal grid lines at 0%, 25%, 50%, 75%, 100%
    for pct in [0, 25, 50, 75, 100]:
        yy = gy + gh - int(gh * pct / 100)
        pygame.draw.line(screen, GRID_COL, (gx, yy), (gx + gw, yy), 1)
        lbl = font_tiny.render(f"{pct}%", True, (100, 100, 130))
        screen.blit(lbl, (gx - 28, yy - 7))

    # 50% reference line
    mid_y = gy + gh // 2
    pygame.draw.line(screen, (80, 80, 110), (gx, mid_y), (gx + gw, mid_y), 1)

    history = list(win_history)
    if len(history) < 2:
        no_data = font_small.render("Waiting for games...", True, (100,100,130))
        screen.blit(no_data, no_data.get_rect(center=(gx + gw//2, gy + gh//2)))
        return

    # Build smoothed win-rate series
    s2_series, s1_series = [], []
    for end in range(1, len(history) + 1):
        chunk = history[:end][-WINDOW:]
        s2_series.append(sum(1 for r in chunk if r == 0) / len(chunk))
        s1_series.append(sum(1 for r in chunk if r == 1) / len(chunk))

    n = len(s2_series)

    def to_screen(idx, rate):
        x = gx + int(idx / max(n - 1, 1) * gw)
        y = gy + gh - int(rate * gh)
        return (x, y)

    # Draw lines
    for series, col in [(s2_series, LINE_S2), (s1_series, LINE_S1)]:
        pts = [to_screen(i, v) for i, v in enumerate(series)]
        if len(pts) >= 2:
            pygame.draw.lines(screen, col, False, pts, 2)

    # Current rate dots
    last_s2 = to_screen(n - 1, s2_series[-1])
    last_s1 = to_screen(n - 1, s1_series[-1])
    pygame.draw.circle(screen, LINE_S2, last_s2, 5)
    pygame.draw.circle(screen, LINE_S1, last_s1, 5)

    # Legend
    leg_y = gy + gh + 14
    pygame.draw.line(screen, LINE_S2, (gx, leg_y + 5), (gx + 20, leg_y + 5), 3)
    screen.blit(font_small.render(f"Stage 2  {s2_series[-1]*100:.0f}%", True, LINE_S2),
                (gx + 24, leg_y - 2))
    pygame.draw.line(screen, LINE_S1, (gx + 130, leg_y + 5), (gx + 150, leg_y + 5), 3)
    screen.blit(font_small.render(f"Stage 1  {s1_series[-1]*100:.0f}%", True, LINE_S1),
                (gx + 154, leg_y - 2))

# -------------------------------------------------------------------
# Run one full game, return winner
# -------------------------------------------------------------------
def run_game(board, current_player):
    while not board.done:
        legal = board.get_legal_moves(current_player)
        if not legal:
            current_player = 1 - current_player
            continue

        obs = board.board.astype(np.float32)

        if current_player == 0:
            action, _ = model_s2.predict(obs, deterministic=True)
        else:
            # Flip perspective for player 1
            flipped = np.concatenate([obs[7:14], obs[0:7]])
            action, _ = model_s1.predict(flipped, deterministic=True)

        action = int(action)
        if action not in legal:
            action = np.random.choice(legal)

        extra, _ = board.make_move(current_player, action)
        if not extra:
            current_player = 1 - current_player

        # Yield for rendering
        yield board.board.copy(), current_player

    return

# -------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------
board_state   = MancalaBoard()
current_player = 0
game_gen      = run_game(board_state, current_player)
step_delay    = 80   # ms between moves (lower = faster)
last_step     = pygame.time.get_ticks()

while True:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                step_delay = max(10, step_delay - 20)
            if event.key == pygame.K_DOWN:
                step_delay = min(500, step_delay + 20)

    now = pygame.time.get_ticks()
    if now - last_step >= step_delay:
        last_step = now
        try:
            snap, current_player = next(game_gen)
        except StopIteration:
            # Game ended
            w = board_state.winner
            win_history.append(w)
            game_count += 1
            if w == 0:   s2_wins += 1
            elif w == 1: s1_wins += 1
            else:        draws   += 1

            # Start new game
            board_state    = MancalaBoard()
            current_player = 0
            game_gen       = run_game(board_state, current_player)

    # Draw
    screen.fill(BG)
    draw_game_board(board_state.board, current_player)
    draw_graph()

    # Speed hint
    hint = font_tiny.render("↑↓ = speed", True, (70, 70, 100))
    screen.blit(hint, (PANEL_X + 8, H - 16))

    pygame.display.flip()
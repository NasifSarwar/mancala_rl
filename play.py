import pygame
import sys
import numpy as np
from stable_baselines3 import PPO
from game.mancala_board import MancalaBoard

# -------------------------------------------------------------------
# Colours
# -------------------------------------------------------------------
BG          = (22,  22,  35)
BOARD_COL   = (40,  40,  62)
PIT_COL     = (70,  70, 105)
PIT_HOVER   = (130, 130, 200)
PIT_LEGAL   = (80,  180, 110)
PIT_LEGAL_H = (110, 220, 140)
STORE_COL   = (32,  32,  52)
TEXT_COL    = (230, 230, 240)
ACCENT      = (180, 120, 255)
AI_COL      = (100, 170, 255)
YOU_COL     = (255, 180,  80)
DIVIDER     = (60,  60,  90)

# -------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------
W, H      = 920, 500
PIT_R     = 44
STORE_W   = 78
STORE_H   = 220
ROW_Y     = [H // 3, int(H * 0.67)]
PIT_XS    = [175 + i * 98 for i in range(6)]
SL        = (68,  H // 2)
SR        = (852, H // 2)

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Mancala — You vs AI")
clock  = pygame.time.Clock()

font_title = pygame.font.SysFont("Georgia", 26, bold=True)
font_big   = pygame.font.SysFont("Georgia", 32, bold=True)
font_med   = pygame.font.SysFont("Georgia", 19)
font_small = pygame.font.SysFont("Georgia", 13)

MODEL_PATH = "models/stage2_selfplay"
model = PPO.load(MODEL_PATH)

board     = MancalaBoard()
HUMAN, AI = 0, 1
message   = "Your turn — click a pit"
game_over = False
hovered   = None

# -------------------------------------------------------------------
def draw_pit(cx, cy, stones, legal=False, hover=False, radius=PIT_R):
    col = PIT_LEGAL_H if hover else (PIT_LEGAL if legal else PIT_COL)
    pygame.draw.circle(screen, col, (cx, cy), radius)
    pygame.draw.circle(screen, (200, 200, 220), (cx, cy), radius, 2)
    t = font_med.render(str(stones), True, TEXT_COL)
    screen.blit(t, t.get_rect(center=(cx, cy)))

def draw_store(cx, cy, stones, label, col):
    r = pygame.Rect(cx - STORE_W//2, cy - STORE_H//2, STORE_W, STORE_H)
    pygame.draw.rect(screen, STORE_COL, r, border_radius=18)
    pygame.draw.rect(screen, col, r, 2, border_radius=18)
    lbl = font_small.render(label, True, col)
    screen.blit(lbl, lbl.get_rect(center=(cx, cy - STORE_H//2 + 18)))
    t = font_big.render(str(stones), True, TEXT_COL)
    screen.blit(t, t.get_rect(center=(cx, cy + 10)))

def draw_board():
    screen.fill(BG)
    card = pygame.Rect(28, H//2 - 175, W - 56, 350)
    pygame.draw.rect(screen, BOARD_COL, card, border_radius=28)
    pygame.draw.line(screen, DIVIDER,
                     (card.left + 20, H//2),
                     (card.right - 20, H//2), 1)

    b = board.board

    # Row labels — left edge of board
    ai_lbl  = font_small.render("AI",  True, AI_COL)
    you_lbl = font_small.render("YOU", True, YOU_COL)
    screen.blit(ai_lbl,  ai_lbl.get_rect(center=(card.left + 28, ROW_Y[0])))
    screen.blit(you_lbl, you_lbl.get_rect(center=(card.left + 28, ROW_Y[1])))

    # AI pits top row
    for i in range(6):
        draw_pit(PIT_XS[i], ROW_Y[0], b[12 - i])

    # Human pits bottom row
    legal = board.get_legal_moves(HUMAN) if not game_over and board.current_player == HUMAN else []
    for i in range(6):
        draw_pit(PIT_XS[i], ROW_Y[1], b[i],
                 legal=(i in legal),
                 hover=(hovered == i and i in legal))

    draw_store(*SL, b[6],  "YOU", YOU_COL)
    draw_store(*SR, b[13], "AI",  AI_COL)

    msg = font_title.render(message, True, ACCENT)
    screen.blit(msg, msg.get_rect(center=(W//2, 28)))

    score = f"You  {b[6]}   :   {b[13]}  AI"
    sc = font_med.render(score, True, TEXT_COL)
    screen.blit(sc, sc.get_rect(center=(W//2, H - 22)))

    hint = font_small.render("R = restart", True, (80, 80, 110))
    screen.blit(hint, (W - 90, H - 18))

    pygame.display.flip()

def get_hovered(mx, my):
    for i in range(6):
        cx, cy = PIT_XS[i], ROW_Y[1]
        if (mx - cx)**2 + (my - cy)**2 <= PIT_R**2:
            return i
    return None

def ai_move():
    global message
    obs     = board.board.astype(np.float32)
    flipped = np.concatenate([obs[7:14], obs[0:7]])
    action, _ = model.predict(flipped, deterministic=True)
    action  = int(action)
    legal   = board.get_legal_moves(AI)
    if action not in legal:
        action = np.random.choice(legal)
    extra, captured = board.make_move(AI, action)
    msg = f"AI played pit {action}"
    if captured: msg += f"  •  captured {captured}"
    if extra:    msg += "  •  extra turn"
    message = msg
    return extra

draw_board()

while True:
    clock.tick(30)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        if event.type == pygame.MOUSEMOTION and not game_over:
            hovered = get_hovered(*event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
            if board.current_player == HUMAN:
                pit = get_hovered(*event.pos)
                if pit is not None and pit in board.get_legal_moves(HUMAN):
                    extra, captured = board.make_move(HUMAN, pit)
                    message = f"You played pit {pit}"
                    if captured: message += f"  •  captured {captured}"
                    if extra:    message += "  •  extra turn!"
                    if not extra:
                        board.current_player = AI
                    draw_board()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            board.reset()
            game_over = False
            message   = "Your turn — click a pit"

    if not game_over and board.current_player == AI and not board.done:
        pygame.time.delay(500)
        extra = ai_move()
        if not extra:
            board.current_player = HUMAN
            if not board.done:
                message = "Your turn — click a pit"

    if board.done and not game_over:
        game_over = True
        w = board.winner
        if w == HUMAN:   message = "You win!  Press R to play again"
        elif w == AI:    message = "AI wins!  Press R to try again"
        else:            message = "Draw!  Press R to play again"

    draw_board()
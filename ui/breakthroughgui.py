import pygame

from games.breakthrough import BreakthroughGameState
from run_games import AIParams, init_ai_player

# Pygame configurations
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
GRID_SIZE = 8
CELL_SIZE = WINDOW_WIDTH // GRID_SIZE

# Colors
WHITE = (245, 245, 220)
BLACK = (47, 79, 79)
BLUE = (70, 130, 180)
RED = (188, 143, 143)
GREEN = (143, 188, 143)

# Pygame window
pygame.init()
WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))


# Pygame window
pygame.init()
WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Breakthrough Game")
FONT = pygame.font.Font(None, 36)


def draw_board(game_state, from_position=None):
    board = game_state.board
    WINDOW.fill(WHITE)

    # Get legal moves
    legal_moves = None
    if from_position is not None:
        legal_moves = [
            to_position for (_, to_position) in game_state.get_legal_actions() if _ == from_position
        ]

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(WINDOW, BLACK, rect, 1)

            pos = i * GRID_SIZE + j

            # Highlight legal moves
            if legal_moves and pos in legal_moves:
                rect = pygame.Rect(j * CELL_SIZE + 1, i * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2)
                pygame.draw.rect(WINDOW, GREEN, rect)

            piece = board[pos]
            if piece != 0:
                color = BLUE if piece == 1 else RED
                pygame.draw.circle(WINDOW, color, rect.center, CELL_SIZE // 2 - 10)

        # Draw chess position notations
        label = FONT.render(str(8 - i), 1, BLACK)
        WINDOW.blit(label, (0, i * CELL_SIZE + 20))
        label = FONT.render(chr(i + 65), 1, BLACK)
        WINDOW.blit(label, (i * CELL_SIZE + 10, 0))

    pygame.display.update()


def main():
    game_state = BreakthroughGameState()
    ai_player = 2
    ai_params = AIParams(
        ai_key="alphabeta",
        eval_params={},
        max_player=2,
        ai_params={"max_depth": 10, "max_time": 10, "debug": True, "use_null_moves": True},
        transposition_table_size=game_state.transposition_table_size,
    )
    ai = init_ai_player(ai_params, BreakthroughGameState.param_order, BreakthroughGameState.default_params)
    from_position = None
    while True:
        ai_to_play = ai_player == game_state.player
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                col, row = x // CELL_SIZE, y // CELL_SIZE
                pos = row * GRID_SIZE + col

                if from_position is None:  # If the 'from' position isn't set, set it
                    from_position = pos
                else:  # If the 'from' position is set, attempt to move the piece
                    action = (from_position, pos)
                    if action in game_state.get_legal_actions():
                        game_state = game_state.apply_action(action)
                    from_position = None  # Reset the 'from' position for the next move

        draw_board(game_state, from_position)

        if ai_to_play:
            # AI player's turn. For now, just pick a random legal action.
            move, _ = ai.best_action(game_state)
            game_state = game_state.apply_action(move)
            print(f"    ---- AI moved {game_state.readable_move(move)} ---- ")

        draw_board(game_state, from_position)


if __name__ == "__main__":
    main()

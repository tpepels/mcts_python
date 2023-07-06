import pygame

from games.amazons import AmazonsGameState
from run_games import AIParams, init_ai_player

# Pygame configurations
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
GRID_SIZE = 10  # change this according to your game board size
CELL_SIZE = WINDOW_WIDTH // GRID_SIZE

# Colors
WHITE = (245, 245, 220)
BLACK = (47, 79, 79)
LIGHT_BLUE = (173, 216, 230)
DARK_RED = (139, 0, 0)
LIGHTER_GREEN = (144, 238, 144)
DARK_GREY = (105, 105, 105)

# Pygame window
pygame.init()
WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Amazons Game")
FONT = pygame.font.Font(None, 36)


def draw_board(game_state: AmazonsGameState, from_position=None, to_position=None):
    board = game_state.board
    WINDOW.fill(WHITE)
    legal_moves = game_state.get_legal_moves_for_amazon(*from_position) if from_position is not None else []
    legal_arrow_shots = (
        [move[4:] for move in legal_moves if move[2:4] == to_position] if to_position is not None else []
    )

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(WINDOW, BLACK, rect, 1)

            if board[i][j] == 1:
                pygame.draw.circle(WINDOW, LIGHT_BLUE, rect.center, CELL_SIZE // 2 - 10)
            elif board[i][j] == 2:
                pygame.draw.circle(WINDOW, DARK_RED, rect.center, CELL_SIZE // 2 - 10)
            elif board[i][j] == -1:
                pygame.draw.rect(WINDOW, DARK_GREY, rect)

            # Highlight legal moves
            if from_position is not None and any(move[2:4] == (i, j) for move in legal_moves):
                pygame.draw.circle(WINDOW, (0, 255, 0), rect.center, CELL_SIZE // 2 - 30)

            # Highlight legal arrow shots
            if to_position is not None and (i, j) in legal_arrow_shots:
                pygame.draw.circle(WINDOW, (255, 0, 0), rect.center, CELL_SIZE // 2 - 30)

        # Draw chess position notations
        label = FONT.render(str(10 - i), 1, BLACK)
        WINDOW.blit(label, (0, i * CELL_SIZE))
        label = FONT.render(chr(i + 65), 1, BLACK)
        WINDOW.blit(label, (i * CELL_SIZE, WINDOW_HEIGHT - CELL_SIZE))

    pygame.display.update()


def main():
    game_state = AmazonsGameState()
    ai_player = 2
    ai_params = AIParams(
        ai_key="alphabeta",
        eval_key="evaluate_amazons_lieberum",
        max_player=2,
        ai_params={"max_depth": 10, "max_time": 10, "debug": True, "use_null_moves": True},
        transposition_table_size=game_state.transposition_table_size,
    )
    ai = init_ai_player(ai_params, ai_player)
    from_position = None
    to_position = None

    while True:
        ai_to_play = ai_player == game_state.player
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                col, row = x // CELL_SIZE, y // CELL_SIZE

                # Check if we're selecting a position to shoot the arrow
                if from_position and to_position:
                    action = (*from_position, *to_position, row, col)
                    print("[3] Attempted action:", action)
                    if action in game_state.get_legal_actions():  # If it's a valid move
                        print("[4] Action executed")
                        game_state = game_state.apply_action(action)
                        from_position = None
                        to_position = None  # Reset the 'from' and 'to' positions for the next move
                # Check if we're selecting a 'to' position
                elif from_position and any(
                    move[2:4] == (row, col) for move in game_state.get_legal_moves_for_amazon(*from_position)
                ):
                    print("[5] Set 'to' position")
                    to_position = (row, col)
                # Check if we're selecting a 'from' position
                elif game_state.board[row][col] == game_state.player:
                    print("[6] Set 'from' position")
                    from_position = (row, col)
                    to_position = None

        draw_board(game_state, from_position, to_position)

        if ai_to_play:
            # AI player's turn. For now, just pick a random legal action.
            move, _ = ai.best_action(game_state)
            game_state = game_state.apply_action(move)
            print(f"    ---- AI moved {game_state.readable_move(move)} ---- ")

        draw_board(game_state, from_position, to_position)


if __name__ == "__main__":
    main()

from games.tictactoe import TicTacToeGameState, MARKS
import pygame
import time

from run_games import AIParams, init_ai_player

# define constants
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 5, 5  # This also determines the board size to play on
SQUARE_SIZE = WIDTH // ROWS

# RGB
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# initialize the game
pygame.init()

# set up the display
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

# set up the clock
CLOCK = pygame.time.Clock()

# set the game title
pygame.display.set_caption("Tic Tac Toe")


def draw_window(state):
    WIN.fill(WHITE)

    # Draw grid
    for i in range(1, ROWS):
        pygame.draw.line(WIN, BLACK, (0, i * SQUARE_SIZE), (WIDTH, i * SQUARE_SIZE))
        for j in range(1, COLS):
            pygame.draw.line(WIN, BLACK, (j * SQUARE_SIZE, 0), (j * SQUARE_SIZE, HEIGHT))

    # Draw marks
    for i in range(ROWS):
        for j in range(COLS):
            mark = MARKS[state.board[i][j]]
            if mark == "X":
                pygame.draw.line(
                    WIN,
                    RED,
                    (j * SQUARE_SIZE + 5, i * SQUARE_SIZE + 5),
                    ((j + 1) * SQUARE_SIZE - 5, (i + 1) * SQUARE_SIZE - 5),
                    10,
                )
                pygame.draw.line(
                    WIN,
                    RED,
                    ((j + 1) * SQUARE_SIZE - 5, i * SQUARE_SIZE + 5),
                    (j * SQUARE_SIZE + 5, (i + 1) * SQUARE_SIZE - 5),
                    10,
                )
            elif mark == "O":
                pygame.draw.circle(
                    WIN,
                    RED,
                    (j * SQUARE_SIZE + SQUARE_SIZE // 2, i * SQUARE_SIZE + SQUARE_SIZE // 2),
                    SQUARE_SIZE // 2 - 5,
                    10,
                )

    pygame.display.update()


def main():
    run = True
    ai_player = 2
    ai_params = AIParams(
        ai_key="alphabeta",
        eval_key="evaluate_tictactoe",
        ai_params={"max_depth": 10, "max_time": 10, "debug": True, "use_null_moves": True},
    )
    ai = init_ai_player(ai_params, ai_player)
    game_state = TicTacToeGameState(board_size=ROWS)

    while run:
        CLOCK.tick(60)
        ai_to_play = ai_player == game_state.player
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.MOUSEBUTTONDOWN and not ai_to_play:
                x, y = pygame.mouse.get_pos()
                row, col = y // SQUARE_SIZE, x // SQUARE_SIZE
                try:
                    game_state = game_state.apply_action((row, col))
                except ValueError:
                    pass  # Illegal move, ignore and let player try again.

        draw_window(game_state)

        if ai_to_play:
            # AI player's turn. For now, just pick a random legal action.
            move, _ = ai.best_action(game_state)
            game_state = game_state.apply_action(move)
            print(f"    ---- AI moved {move} ---- ")

        draw_window(game_state)

        if game_state.is_terminal():
            print(game_state.get_reward(ai_player))
            time.sleep(4)
            game_state = TicTacToeGameState()  # Reset the game state.

    pygame.quit()


if __name__ == "__main__":
    main()

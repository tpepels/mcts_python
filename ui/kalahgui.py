import time
import pygame
from games.kalah import KalahGameState
from run_games import AIParams, init_ai_player

# define constants
WIDTH, HEIGHT = 800, 200
PIT_SIZE = WIDTH // 14
SEED_SIZE = PIT_SIZE // 8

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
pygame.display.set_caption("Kalah")


def draw_window(state):
    WIN.fill(WHITE)

    # Prepare font for rendering the number of seeds
    font = pygame.font.Font(None, PIT_SIZE // 2)  # Adjust font size according to your preference

    # Draw pits
    for i in range(14):
        if i == 13:  # Player 1's home pit
            x = PIT_SIZE // 2
            y = HEIGHT // 2
        elif i == 6:  # Player 2's home pit
            x = 7 * PIT_SIZE + PIT_SIZE // 2
            y = HEIGHT // 2
        elif i < 6:
            x = (i + 1) * PIT_SIZE + PIT_SIZE // 2
            y = 3 * HEIGHT // 4
        else:
            x = (13 - i) * PIT_SIZE + PIT_SIZE // 2
            y = HEIGHT // 4

        pygame.draw.circle(WIN, BLACK, (x, y), PIT_SIZE // 2, 3)

        # Render the number of seeds and display it
        seeds_num = state.board[i]
        text = font.render(str(seeds_num), True, BLACK)
        text_rect = text.get_rect(center=(x, y))
        WIN.blit(text, text_rect)

    pygame.display.update()


def main():
    run = True
    game_state = KalahGameState()
    ai_player = 2
    ai_params = AIParams(
        ai_key="alphabeta",
        eval_key="evaluate_kalah",
        ai_params={"max_depth": 10, "max_time": 10, "debug": True, "use_null_moves": True},
    )
    ai = init_ai_player(ai_params, ai_player)

    while run:
        CLOCK.tick(60)
        ai_to_play = ai_player == game_state.player
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()

                # Calculate the pit index based on mouse position
                if y > HEIGHT // 2:
                    if x < PIT_SIZE:  # Player 1's home pit
                        pit_index = 6
                    else:
                        pit_index = x // PIT_SIZE - 1
                else:
                    if x > 7 * PIT_SIZE:  # Player 2's home pit
                        pit_index = 13
                    else:
                        pit_index = 13 - x // PIT_SIZE

                if game_state._is_valid_move(pit_index):
                    game_state = game_state.apply_action(pit_index)

        draw_window(game_state)

        if ai_to_play:
            # AI player's turn. For now, just pick a random legal action.
            move, _ = ai.best_action(game_state)
            game_state = game_state.apply_action(move)
            print(f"    ---- AI moved {move} ---- ")

        draw_window(game_state)

        if game_state.is_terminal():
            time.sleep(4)
            game_state = KalahGameState()  # Reset the game state.

    pygame.quit()


if __name__ == "__main__":
    main()

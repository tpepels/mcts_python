import pygame
from breakthroughgui import BLUE

from games.minishogi import MiniShogi
from pygame.locals import *
from run_games import AIParams, init_ai_player

from ui.kalahgui import RED

# Initialize Pygame
pygame.init()

from minishogi_definitions import *

# Load the piece images used by the funcions below
piece_rects = load_images()

# Set transparency and color for the king highlight
king_highlight_color_black = (200, 100, 100, 128)
king_highlight_color_white = (100, 100, 200, 128)


# Function to draw a transparent circle
def draw_transparent_circle(surface, color, center, radius):
    target_rect = pygame.Rect(center, (0, 0)).inflate((radius * 2, radius * 2))
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.circle(shape_surf, color, (radius, radius), radius)
    surface.blit(shape_surf, target_rect.topleft)


def handle_captured_piece_click(mouse_x, mouse_y, captured_pieces_1, captured_pieces_2, player):
    label_height = font_size + 10  # The same adjustment used in draw_captured_pieces

    # Define the areas for player 1 and player 2's captured pieces based on the UI layout
    captured_area_p1_start = 0
    captured_area_p1_end = captured_area_width
    captured_area_p2_start = screen_width - captured_area_width
    captured_area_p2_end = screen_width

    captured_piece = None

    if captured_area_p1_start <= mouse_x <= captured_area_p1_end:
        # Click is within Player 1's captured pieces area
        index = (mouse_y - label_height) // square_size
        if 0 <= index < len(captured_pieces_1):
            captured_piece = captured_pieces_1[index]
            player = 1
    elif captured_area_p2_start <= mouse_x <= captured_area_p2_end:
        # Click is within Player 2's captured pieces area
        index = (mouse_y - label_height) // square_size
        if 0 <= index < len(captured_pieces_2):
            captured_piece = captured_pieces_2[index]
            player = 2

    if captured_piece is not None:
        # Return the captured piece and the player it belongs to
        print("Clicked on captured piece:", captured_piece, "Player:", player)
        return captured_piece, player
    else:
        # Return None if no captured piece was clicked
        return None


# Function to draw the captured pieces
def draw_captured_pieces(screen, captured_pieces_1, captured_pieces_2, square_size):
    label_height = font_size + 10  # Additional space above the label
    # Draw labels for captured pieces areas
    label_p1 = font.render("Captured P1", True, BLACK)
    label_p2 = font.render("Captured P2", True, BLACK)
    screen.blit(label_p1, (0, 0))  # Top left for Player 1
    screen.blit(label_p2, (screen_width - captured_area_width, 0))  # Top right for Player 2

    # Draw Player 1's captured pieces on the left side
    for i, piece in enumerate(captured_pieces_1):
        piece_rect = pygame.Rect(0, i * square_size + label_height, square_size, square_size)
        draw_piece(screen, piece, piece_rect.topleft)

    # Draw Player 2's captured pieces on the right side
    for i, piece in enumerate(captured_pieces_2):
        piece_rect = pygame.Rect(
            screen_width - captured_area_width, i * square_size + label_height, square_size, square_size
        )
        draw_piece(screen, piece, piece_rect.topleft)


all_legal_actios = {}


def draw_legal_moves(screen, row, col, font):
    legal_actions = get_legal_actions_for_piece(game_state, row, col)
    for action in legal_actions:
        # For moves, action is a tuple (from_row, from_col, to_row, to_col)
        if action[0] != -1:
            evaluation = game_state.evaluate_move(action)  # Evaluate the move
            center_x = captured_area_width + action[3] * square_size + square_size // 2
            center_y = action[2] * square_size + square_size // 2
            center = (center_x, center_y)
            pygame.draw.circle(screen, GREEN, center, square_size // 10)
            eval_text = font.render(str(evaluation), True, BLACK)  # Render the evaluation as text
            screen.blit(eval_text, center)  # Draw the evaluation text near or inside the circle


def draw_legal_drops(screen, piece_id, player, font):
    legal_actions = game_state.get_legal_actions()  # This needs to return actions including drops
    for action in legal_actions:
        if action[0] == -1 and action[1] == piece_id:
            evaluation = game_state.evaluate_move(action)  # Evaluate the move
            # This is a legal drop action for the selected piece
            center_x = captured_area_width + action[3] * square_size + square_size // 2
            center_y = action[2] * square_size + square_size // 2
            rect = pygame.Rect(center_x - square_size // 2, center_y - square_size // 2, square_size, square_size)
            pygame.draw.rect(screen, BLUE, rect, 2)
            eval_text = font.render(str(evaluation), True, BLACK)  # Render the evaluation as text
            screen.blit(eval_text, rect.topleft)  # Adjust the position as needed


# Add to your MiniShogi class
def get_legal_actions_for_piece(state, row, col):
    # Assuming a function that gets all legal actions for a specific piece
    all_legal_actions = state.get_legal_actions()
    # Filter for actions that start from the given row and col
    return [action for action in all_legal_actions if action[:2] == (row, col)]


def draw_piece(screen, piece_id, position):
    # Assuming get_piece_image returns a Pygame Surface for the given piece ID
    piece_image = get_piece_image(piece_rects[piece_id])

    # Calculate offsets to center the piece image within its square
    image_offset_x = (square_size - piece_image.get_width()) // 2
    image_offset_y = (square_size - piece_image.get_height()) // 2

    # Draw the piece image centered within the square
    screen.blit(piece_image, (position[0] + image_offset_x, position[1] + image_offset_y))

    # Optionally, draw the character for the piece below the image for clarity
    piece_char = PIECE_CHARS.get(piece_id, "")

    if 0 < piece_id <= 10:
        # Calculate position to draw the PIECE_CHAR below the piece image
        char_position = (position[0] + 5, position[1] + 5)
    else:
        char_position = (position[0] + 5, position[1] + square_size - 20)

    font = pygame.font.SysFont(None, 24)
    text_surf = font.render(piece_char, True, BLACK)  # BLACK is the text color
    screen.blit(text_surf, char_position)


# Colors
WHITE = (245, 245, 220)
BLACK = (47, 79, 79)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# Set up the display
square_size = 100  # Square size for each chess piece
board_width = square_size * 5  # Board width for a 5x5 board
captured_area_width = square_size * 2  # Width for the area to display captured pieces
screen_width = board_width + 2 * captured_area_width  # Total screen width including areas for captured pieces
screen_height = square_size * 5 + 50  # Adding 50 pixels space for displaying player and check status

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Minishogi")

game_state = MiniShogi()
print(game_state.visualize(full_debug=True))

# Set up the font for drawing labels
pygame.font.init()  # Initialize the font module
font_size = 24
font = pygame.font.SysFont("Arial", font_size)
ab_params = AIParams(
    ai_key="alphabeta",
    eval_params={},
    max_player=2,
    game_name="minishogi",
    ai_params={"max_depth": 10, "max_time": 10, "debug": True},
    transposition_table_size=game_state.transposition_table_size,
)
ab_ai = init_ai_player(ab_params, MiniShogi.param_order, MiniShogi.default_params)

mcts_player = 1
mcts_params = AIParams(
    ai_key="mcts",
    eval_params={},
    max_player=1,
    game_name="minishogi",
    ai_params={"max_time": 10, "early_term_turns": 5, "debug": True},
    transposition_table_size=game_state.transposition_table_size,
)

mcts_ai = init_ai_player(mcts_params, MiniShogi.param_order, MiniShogi.default_params)

# Main game loop
running = True
selected_piece_pos = None
selected_captured_piece = None
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            col = (mouse_x - captured_area_width) // square_size
            row = mouse_y // square_size

            # First, check if the click is within the captured pieces area
            captured_piece_info = handle_captured_piece_click(
                mouse_x,
                mouse_y,
                game_state.captured_pieces_1,
                game_state.captured_pieces_2,
                game_state.player,
            )

            if captured_piece_info:
                # If a captured piece was clicked, store the selection for dropping
                selected_captured_piece = captured_piece_info
                selected_piece_pos = None  # Clear any board selection
                continue  # Skip further checks

            if not game_state.is_terminal() and (0 <= row < 5 and 0 <= col < 5):  # Ensure click is within the board
                piece_id = game_state.board[row][col]
                # Determine if the clicked piece belongs to the current player
                belongs_to_current_player = (game_state.player == 1 and 1 <= piece_id <= 10) or (
                    game_state.player == 2 and 11 <= piece_id <= 20
                )

                if selected_captured_piece and game_state.board[row][col] == 0:
                    # Try to drop the captured piece if one is selected
                    piece_id, player = selected_captured_piece
                    action = (-1, piece_id, row, col)
                    if action in game_state.get_legal_actions():
                        game_state = game_state.apply_action(action)
                        selected_captured_piece = None  # Clear captured piece selection after drop
                        print(game_state.visualize(full_debug=True))

                # Selecting a different piece or deselecting the current piece
                elif belongs_to_current_player and selected_piece_pos != (row, col):
                    # Select the current player's piece for moving
                    selected_piece_pos = (row, col)
                    selected_captured_piece = None  # Clear captured piece selection
                elif selected_piece_pos:
                    # Try to move, promote or capture
                    from_row, from_col = selected_piece_pos
                    action = (from_row, from_col, row, col)
                    if action in game_state.get_legal_actions():
                        game_state = game_state.apply_action(action)
                        print(game_state.visualize(full_debug=True))

                    selected_piece_pos = None  # Clear selection after move or capture

    screen.fill(WHITE)
    # Draw the board and pieces
    for row in range(5):
        for col in range(5):
            square_rect = pygame.Rect(
                captured_area_width + col * square_size, row * square_size, square_size, square_size
            )
            pygame.draw.rect(screen, BLACK, square_rect, 1)
            # Draw coordinates
            coord_text = chr(97 + col) + str(5 - row)  # Convert to 'a' - 'e' for columns and 1-5 for rows
            text_surface = pygame.font.SysFont("Arial", 16).render(coord_text, True, BLACK)
            screen.blit(text_surface, (square_rect.left + 5, square_rect.top + 5))

            piece_id = game_state.board[row][col]
            if piece_id != 0:
                draw_piece(screen, piece_id, square_rect.topleft)

    # Draw captured pieces for both players
    draw_captured_pieces(screen, game_state.captured_pieces_1, game_state.captured_pieces_2, square_size)

    # Draw legal moves if a piece is selected
    if selected_piece_pos:
        selected_piece_rect = pygame.Rect(
            captured_area_width + selected_piece_pos[1] * square_size,
            selected_piece_pos[0] * square_size,
            square_size,
            square_size,
        )
        pygame.draw.rect(screen, GREEN, selected_piece_rect, 2)  # Highlight selected piece
        draw_legal_moves(screen, *selected_piece_pos, font)

    if selected_captured_piece:
        piece_id, player = selected_captured_piece
        # Draw all legal drops on the board
        draw_legal_drops(screen, piece_id, player, font)

    # Display the current player
    current_player_text = "Player to move: Black" if game_state.player == 1 else "Player to move: White"
    text_surface = font.render(current_player_text, True, BLACK)
    screen.blit(text_surface, (10, screen_height - 45))

    # Highlight the king positions
    king_1_pos = game_state.king_1
    king_2_pos = game_state.king_2
    king_1_center = (
        captured_area_width + king_1_pos[1] * square_size + square_size // 2,
        king_1_pos[0] * square_size + square_size // 2,
    )
    king_2_center = (
        captured_area_width + king_2_pos[1] * square_size + square_size // 2,
        king_2_pos[0] * square_size + square_size // 2,
    )

    draw_transparent_circle(
        screen, king_highlight_color_black, king_1_center, square_size // 4
    )  # Highlight for Player 1's king
    draw_transparent_circle(
        screen, king_highlight_color_white, king_2_center, square_size // 4
    )  # Highlight for Player 2's king

    # Display if the king is under check
    if game_state.is_terminal():
        check_text = "Checkmate!"
        check_surface = font.render(check_text, True, RED)
        screen.blit(check_surface, (screen_width - 150, screen_height - 45))
    elif game_state.check:
        check_text = "Check!"
        check_surface = font.render(check_text, True, RED)
        screen.blit(check_surface, (screen_width - 150, screen_height - 45))

    if not game_state.is_terminal() and game_state.player == 1:
        move, _ = mcts_ai.best_action(game_state)
        game_state = game_state.apply_action(move)
        print("mcts action applied:", move)
        # Optional: Print the game state or any other debug information after the random move
        print(game_state.visualize(full_debug=True))

    if not game_state.is_terminal() and game_state.player == 2:
        move, _ = ab_ai.best_action(game_state)
        game_state = game_state.apply_action(move)
        print("alpha beta action applied:", move)
        # Optional: Print the game state or any other debug information after the random move
        print(game_state.visualize(full_debug=True))

    pygame.display.flip()

pygame.quit()

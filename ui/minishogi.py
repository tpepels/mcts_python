import pygame
from breakthroughgui import BLUE

from games.minishogi import MiniShogi
from pygame.locals import *

from ui.kalahgui import RED

# Initialize Pygame
pygame.init()

from minishogi_definitions import *

# Load the piece images used by the funcions below
piece_rects = load_images()


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


def draw_legal_moves(screen, row, col):
    legal_actions = get_legal_actions_for_piece(game_state, row, col)
    print(legal_actions)
    for action in legal_actions:
        # For moves, action is a tuple (from_row, from_col, to_row, to_col)
        if action[0] != -1:
            center_x = captured_area_width + action[3] * square_size + square_size // 2
            center_y = action[2] * square_size + square_size // 2
            center = (center_x, center_y)
            pygame.draw.circle(screen, GREEN, center, square_size // 10)


def draw_legal_captures(screen, piece_id, player):
    legal_actions = game_state.get_legal_actions()  # This needs to return actions including drops
    for action in legal_actions:
        if action[0] == -1 and action[1] == piece_id:
            # This is a legal drop action for the selected piece
            center_x = captured_area_width + action[3] * square_size + square_size // 2
            center_y = action[2] * square_size + square_size // 2
            pygame.draw.rect(
                screen, BLUE, (center_x - square_size // 2, center_y - square_size // 2, square_size, square_size), 2
            )


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
print(game_state.is_terminal())
print(game_state.get_reward(1))

# Set up the font for drawing labels
pygame.font.init()  # Initialize the font module
font_size = 24
font = pygame.font.SysFont("Arial", font_size)


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

            if not game_state.is_terminal() and 0 <= row < 5 and 0 <= col < 5:  # Ensure click is within the board
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
                    selected_piece_pos = None  # Clear selection after move or capture

    screen.fill(WHITE)
    # Draw the board and pieces
    for row in range(5):
        for col in range(5):
            square_rect = pygame.Rect(
                captured_area_width + col * square_size, row * square_size, square_size, square_size
            )
            pygame.draw.rect(screen, BLACK, square_rect, 1)
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
        draw_legal_moves(screen, *selected_piece_pos)

    if selected_captured_piece:
        piece_id, player = selected_captured_piece
        # Draw all legal drops on the board
        draw_legal_captures(screen, piece_id, player)

    # Display the current player
    current_player_text = "Player to move: Black" if game_state.player == 1 else "Player to move: White"
    text_surface = font.render(current_player_text, True, BLACK)
    screen.blit(text_surface, (10, screen_height - 45))

    # Display if the king is under check
    if game_state.check:
        check_text = "Check!"
        check_surface = font.render(check_text, True, RED)
        screen.blit(check_surface, (screen_width - 150, screen_height - 45))

    if game_state.is_terminal():
        check_text = "Checkmate!"
        check_surface = font.render(check_text, True, RED)
        screen.blit(check_surface, (screen_width - 150, screen_height - 45))

    pygame.display.flip()

pygame.quit()

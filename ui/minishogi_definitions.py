import pygame


# The dimensions of each piece image within the larger image
piece_width = 81
piece_height = 90
# The number of pieces per player and per row
pieces_per_row = 8
rows_per_player = 2
# Load the image containing all the pieces
pieces_image = pygame.image.load("ui/mono_648.png")

PIECE_CHARS = {
    1: "K",
    2: "G",  # Regular gold general (cannot be promoted)
    3: "P",
    4: "+P",  # An upgraded pawn is a gold general
    5: "S",
    6: "+S",  # A promoted silver general is a gold general
    7: "R",
    8: "+R",  # A promoted rook is a dragon king
    9: "B",
    10: "+B",  # A promoted bishop is a dragon horse
    11: "K",
    12: "G",  # Regular gold general (cannot be promoted)
    13: "P",
    14: "+P",  # An upgraded pawn is a gold general
    15: "S",
    16: "+S",  # A promoted silver general is a gold general
    17: "R",
    18: "+R",  # A promoted rook is a dragon king
    19: "B",
    20: "+B",  # A promoted bishop is a dragon horse
}


# Function to extract and return a piece image from the sprite sheet
def get_piece_image(rect):
    piece_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    piece_surf.blit(pieces_image, (0, 0), rect)
    return piece_surf


def load_images():
    # Create the piece_rects dictionary
    piece_rects = [{}, {}]  # One dictionary for each player

    for player in range(2):  # Two players, 0 and 1
        for row in range(rows_per_player):  # Two rows per player
            for col in range(pieces_per_row):  # 9 pieces per row
                piece_name = row * pieces_per_row + col
                x = col * piece_width
                y = (player * rows_per_player + row) * piece_height
                piece_rects[player][piece_name] = pygame.Rect(x, y, piece_width, piece_height)

    # Map piece names to internal numbers
    PIECES_MAPPING = {
        8: 1,
        3: 2,  # Regular gold general (cannot be promoted)
        7: 3,
        15: 4,  # An upgraded pawn is a gold general
        4: 5,
        12: 6,  # A promoted silver general is a gold general
        1: 7,
        9: 8,  # A promoted rook is a dragon king
        2: 9,
        10: 10,  # A promoted bishop is a dragon horse
    }
    mapped_piece_rects = [0] * 21  # One dictionary for each player
    # Remap the piece names to internal numbers
    for k, v in PIECES_MAPPING.items():
        mapped_piece_rects[v] = piece_rects[0][k]
        mapped_piece_rects[10 + v] = piece_rects[1][k]

    return mapped_piece_rects

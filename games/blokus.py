import copy
from typing import List, Tuple
import numpy as np
from termcolor import colored

from games.gamestate import GameState, win, loss, draw

PLAYER_COUNT = 2
BOARD_SIZE = 20

PASS_MOVE = (-1, -1, -1, -1)
START_POSITIONS = [
    (0, 0),
    (0, BOARD_SIZE - 1),
    (BOARD_SIZE - 1, 0),
    (BOARD_SIZE - 1, BOARD_SIZE - 1),
]  # corner positions

DIAGONALS = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
# Define offsets for orthogonally adjacent cells (up, down, left, right)
ORTHOGONAL_NEIGHBORS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
SIDES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CORNERS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
# We create a constant player board as it's just used for comparing against the actual board
PLAYER_BOARD = [np.full((BOARD_SIZE, BOARD_SIZE), 1), np.full((BOARD_SIZE, BOARD_SIZE), 2)]

# Pre-compute the center weights
center_coord = np.array([BOARD_SIZE, BOARD_SIZE]) // 2
distances = np.abs(np.indices((BOARD_SIZE, BOARD_SIZE)).T - center_coord)
CENTER_WEIGHTS = np.sqrt(np.sum(distances**2, axis=-1))
max_distance = np.sqrt(2 * (BOARD_SIZE // 2) ** 2)

CENTER_WEIGHTS = np.round(max_distance - CENTER_WEIGHTS)
NEIGHBORS = (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)

BOARD_CORNERS = [
    (0, 0),
    (0, BOARD_SIZE - 1),
    (BOARD_SIZE - 1, 0),
    (BOARD_SIZE - 1, BOARD_SIZE - 1),
]


pieces = [
    np.array([[1]]),  # monomino
    # --
    np.array([[1, 1]]),  # domino
    # --
    np.array([[1, 1, 1]]),  # tromino_I
    np.array([[1, 0], [1, 1]]),  # tromino_L
    # --
    np.array([[1, 1, 1, 1]]),  # tetromino_I
    np.array([[1, 0], [1, 0], [1, 1]]),  # tetromino_L
    np.array([[1, 1, 1], [0, 1, 0]]),  # tetromino_T
    np.array([[1, 1, 0], [0, 1, 1]]),  # tetromino_S
    np.array([[1, 1], [1, 1]]),  # tetromino_O
    # --
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]]),  # pentomino_V
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]]),  # pentomino_F
    np.array([[1, 1, 1, 1, 1]]),  # pentomino_I
    np.array([[1, 0], [1, 0], [1, 0], [1, 1]]),  # pentomino_L
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]]),  # pentomino_N
    np.array([[1, 1], [1, 1], [1, 0]]),  # pentomino_P
    np.array([[1, 1], [1, 0], [1, 1]]),  # pentomino_U
    np.array([[1, 0], [1, 0], [1, 1], [0, 1]]),  # pentomino_J
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]]),  # pentomino_T
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]]),  # pentomino_W
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),  # pentomino_X
    np.array([[1, 0], [1, 1], [1, 0], [1, 0]]),  # pentomino_Y
]
# Calculate the total size of the pieces
total_ones = 0
for piece in pieces:
    total_ones += np.sum(piece)


unique_rotations = {}

for i, piece in enumerate(pieces):
    rotations = [np.rot90(piece, rotation) for rotation in range(4)]
    unique = [piece.tolist()]  # initialize with original piece

    for rotated in rotations:
        if not any(np.array_equal(rotated, np.array(u)) for u in unique):
            unique.append(rotated.tolist())

    unique_rotations[i] = len(unique)

MAX_PIECE_COUNT = total_ones * 2


class BlokusPieces:
    zobrist_pieces = np.random.randint(
        low=0,
        high=np.iinfo(np.uint64).max,
        size=(2, len(pieces), 3),  # Player count, number of pieces, piece count (0, 1, or 2)
        dtype=np.uint64,
    )

    def __init__(self):
        self.available_pieces = {(player, i): 2 for player in [1, 2] for i in range(len(pieces))}
        self.pieces_count = [self._piece_count(player) for player in [1, 2]]
        self.pieces_hash = self.calculate_hash()

    def calculate_hash(self):
        hash_value = np.uint64(0)
        for player, piece_index in self.available_pieces:
            hash_value ^= self.zobrist_pieces[
                player - 1, piece_index, self.available_pieces[(player, piece_index)]
            ]
        return hash_value

    def copy(self):
        return copy.deepcopy(self)

    def rotate_piece(self, piece):
        return np.rot90(piece)

    def play_piece(self, piece_index, player):
        key = (player, piece_index)
        piece_count = self.available_pieces[key]
        if piece_count > 0:
            self.available_pieces[key] -= 1
            self.pieces_hash ^= self.zobrist_pieces[player - 1, piece_index, piece_count]
            self.pieces_hash ^= self.zobrist_pieces[player - 1, piece_index, piece_count - 1]
            self.pieces_count[player - 1] -= 1
        else:
            raise ValueError(f"Player {player} doesn't have piece {piece_index} available.")

    def get_available_pieces(self, player):
        return [
            piece_index
            for (player_index, piece_index), count in self.available_pieces.items()
            if count > 0 and player_index == player
        ]

    def pieces_left_for_player(self, player):
        return self.pieces_count[player - 1]

    def _piece_count(self, player) -> int:
        return sum(
            count for (player_index, _), count in self.available_pieces.items() if player_index == player
        )

    def __str__(self):
        pieces_str = ""
        for (player, piece_index), count in self.available_pieces.items():
            pieces_str += f"Player {player}, Piece {piece_index}: {count}\n"
        return (
            f"BlokusPieces:\n"
            f"Available pieces (Player, Piece: Count):\n"
            # f"{pieces_str}"
            f"Pieces count: {self.pieces_count}\n"
            f"Pieces hash: {self.pieces_hash}"
        )


class BlokusGameState(GameState):
    zobrist_table = np.random.randint(
        low=0,
        high=np.iinfo(np.uint64).max,
        size=(BOARD_SIZE, BOARD_SIZE, len(pieces)),
        dtype=np.uint64,
    )
    zobrist_player = np.random.randint(
        low=0, high=np.iinfo(np.uint64).max, size=(PLAYER_COUNT,), dtype=np.uint64
    )

    def __init__(self, board=None, pieces=None, player=1, n_turns=0, passes=0, perimeters=None):
        if pieces is None:
            self.pieces = BlokusPieces()
        else:
            self.pieces = pieces

        self.player = player

        if board is not None:
            self.board = board
            self.perimeters = perimeters
        else:
            self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
            self.board_hash = (
                self._calculate_board_hash() ^ self.zobrist_player[self.player - 1] ^ self.pieces.pieces_hash
            )
            # Initialize the perimeters as a dictionary with player numbers as keys
            self.perimeters = {1: set(BOARD_CORNERS.copy()), 2: set(BOARD_CORNERS.copy())}

        self.n_turns = n_turns
        self.passes = passes
        self.positions_checked = 0

    def _calculate_board_hash(self):
        board_hash = np.uint64(0)
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if self.board[x, y] != 0:
                    player = int(self.board[x, y])
                    board_hash ^= self.zobrist_table[x, y, player - 1]
        return board_hash

    def skip_turn(self):
        """Used for the null-move heuristic in alpha-beta search"""
        new_state = BlokusGameState(
            board=np.copy(self.board),
            pieces=self.pieces.copy(),
            player=3 - self.player,
            n_turns=self.n_turns,
            passes=self.passes + 1,
            perimeters=self.perimeters.copy(),
        )
        new_state.board_hash = (
            new_state._calculate_board_hash()
            ^ self.zobrist_player[new_state.player - 1]
            ^ new_state.pieces.pieces_hash
        )
        return new_state

    def apply_action(self, action: Tuple[int, int, int, int]) -> "BlokusGameState":
        new_state = BlokusGameState(
            board=np.copy(self.board),
            pieces=self.pieces.copy(),
            player=self.player,
            n_turns=self.n_turns,
            passes=self.passes,
            perimeters=self.perimeters.copy(),
        )
        # Pass move if player cannot make a move anymore
        if action == PASS_MOVE:
            new_state.passes += 1
        else:
            x, y, piece_index, rotation = action
            piece = np.rot90(pieces[piece_index], rotation)
            piece_width, piece_height = piece.shape

            # Update the piece to contain the player number instead of True, leave False as is (empty squares)
            player_piece = np.where(piece, self.player, 0)

            # Create a mask of the area on the board where the piece will be placed
            board_mask = new_state.board[x : x + piece_width, y : y + piece_height].astype(bool)
            # Combine the piece mask with the board mask to get the final mask for the update
            update_mask = np.logical_or(board_mask, piece)

            # Create an updated board segment using where the mask is True
            updated_board_segment = np.where(
                update_mask, player_piece, new_state.board[x : x + piece_width, y : y + piece_height]
            )

            # Place the updated board segment back into the board
            new_state.board[x : x + piece_width, y : y + piece_height] = updated_board_segment

            # After placing the piece, update the perimeter
            piece_mask = piece.astype(bool)

            # Remove cells where the new piece was placed from the perimeter
            new_state.perimeters[self.player] -= set(
                (x + i, y + j) for i in range(piece_width) for j in range(piece_height) if piece_mask[i, j]
            )

            # Remove cells that are now orthogonally adjacent to the piece
            for i in range(piece_width):
                for j in range(piece_height):
                    if piece_mask[i, j]:  # For each cell that is part of the piece
                        for dx, dy in ORTHOGONAL_NEIGHBORS:
                            nx, ny = x + i + dx, y + j + dy
                            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                                if (nx, ny) in new_state.perimeters[self.player]:
                                    new_state.perimeters[self.player].remove((nx, ny))

            # Add neighboring cells of the new piece that are empty and not already in the perimeter
            for i in range(piece_width):
                for j in range(piece_height):
                    if piece_mask[i, j]:  # For each cell that is part of the piece
                        for dx, dy in NEIGHBORS:
                            nx, ny = x + i + dx, y + j + dy
                            # Only add the cell to the perimeter if it's not orthogonally adjacent to any piece
                            if (
                                0 <= nx < BOARD_SIZE
                                and 0 <= ny < BOARD_SIZE
                                and new_state.board[nx, ny] == 0
                                and all(
                                    new_state.board[nx + dx, ny + dy] == 0
                                    for dx, dy in ORTHOGONAL_NEIGHBORS
                                    if 0 <= nx + dx < BOARD_SIZE and 0 <= ny + dy < BOARD_SIZE
                                )
                            ):
                                new_state.perimeters[self.player].add((nx, ny))

            new_state.n_turns += 1

            new_state.pieces.play_piece(piece_index, self.player)

        new_state.player = 3 - self.player

        new_state.board_hash = (
            new_state._calculate_board_hash()
            ^ self.zobrist_player[new_state.player - 1]
            ^ new_state.pieces.pieces_hash
        )
        # print(f"New board hash: {new_state.board_hash}")
        return new_state

    def get_legal_actions(self) -> List[Tuple[int, int, int, int]]:
        self.positions_checked = 0
        legal_actions = []

        for position in self.perimeters[self.player]:
            for piece_index in self.pieces.get_available_pieces(self.player):
                for rotation in range(unique_rotations[piece_index]):  # Use pre-computed rotations
                    if self.is_legal_action((*position, piece_index, rotation)):
                        legal_actions.append((*position, piece_index, rotation))

        if not legal_actions:  # If legal_actions list is empty, then it's a PASS_MOVE
            return [PASS_MOVE]

        return legal_actions

    def is_legal_action(self, action: Tuple[int, int, int, int]) -> bool:
        self.positions_checked += 1
        x, y, piece_index, rotation = action
        if rotation > 0:
            piece = np.rot90(pieces[piece_index], rotation)
        else:
            piece = pieces[piece_index]

        piece_mask = piece.astype(bool)
        piece_width, piece_height = piece.shape

        # Check if piece can be physically placed, considering its shape
        if not (
            0 <= x < BOARD_SIZE
            and 0 <= y < BOARD_SIZE
            and 0 <= x + piece_width - 1 < BOARD_SIZE
            and 0 <= y + piece_height - 1 < BOARD_SIZE
        ):
            return False

        board_segment = self.board[x : x + piece_width, y : y + piece_height]
        board_mask = board_segment.astype(bool)

        # Verify if it's the first piece and if it's on the corner
        if self.n_turns <= 3 and self.is_on_corner(x, y, piece):
            return not np.any(
                np.logical_and(piece_mask, board_mask)
            )  # Piece should not overlap with an existing piece

        elif self.n_turns <= 1:
            return False

        return self.is_valid_placement(x, y, piece_mask)

    def is_valid_placement(self, x: int, y: int, piece_mask: np.array) -> bool:
        # board_segment = self.board[x : x + piece_mask.shape[0], y : y + piece_mask.shape[1]]

        # Overlapping thing
        for dx, dy in np.argwhere(piece_mask):
            if (x + dx, y + dy) not in self.perimeters[self.player]:
                return False

        # Check corner cells
        corner_touch = any((x + dx, y + dy) in self.perimeters[self.player] for dx, dy in CORNERS)

        # If no corner is touching the own cells, return False
        if not corner_touch:
            return False

        return True

        # def get_legal_actions(self) -> List[Tuple[int, int, int, int]]:

    #     self.positions_checked = 0
    #     legal_actions = []

    #     if self.n_turns <= 1:  # For first two turns
    #         for piece_index in self.pieces.get_available_pieces(self.player):
    #             for rotation in range(unique_rotations[piece_index]):  # Use pre-computed rotations
    #                 if rotation > 0:
    #                     rotated_piece = np.rot90(pieces[piece_index], rotation)
    #                 else:
    #                     rotated_piece = pieces[piece_index]
    #                 piece_width, piece_height = rotated_piece.shape
    #                 for sx, sy in START_POSITIONS:
    #                     for x in range(sx, sx - piece_width, -1):
    #                         for y in range(sy, sy - piece_height, -1):
    #                             if self.is_legal_action((x, y, piece_index, rotation)):
    #                                 legal_actions.append((x, y, piece_index, rotation))
    #     else:  # For all other turns
    #         # Get a list of the positions of the player's pieces on the board
    #         player_pieces_positions = np.array(np.where(self.board == self.player)).T

    #         # Create offsets for neighboring cells
    #         offsets = np.mgrid[-1:2, -1:2].T.reshape(-1, 2)

    #         # Apply offsets to each piece's position and clamp within board size
    #         neighbors = np.clip(player_pieces_positions[:, None] + offsets[None, :], 0, BOARD_SIZE - 1)

    #         # Flatten and get unique positions
    #         unique_positions = np.unique(neighbors.reshape(-1, 2), axis=0)

    #         potential_positions = set()
    #         for piece_index in self.pieces.get_available_pieces(self.player):
    #             for px, py in unique_positions:  # Iterate over the unique positions
    #                 for rotation in range(unique_rotations[piece_index]):  # Use pre-computed rotations
    #                     if rotation > 0:
    #                         rotated_piece = np.rot90(pieces[piece_index], rotation)
    #                     else:
    #                         rotated_piece = pieces[piece_index]

    #                     piece_width, piece_height = rotated_piece.shape

    #                     for x in range(max(0, px - piece_width + 1), min(BOARD_SIZE, px + piece_width)):
    #                         for y in range(max(0, py - piece_height + 1), min(BOARD_SIZE, py + piece_height)):
    #                             potential_positions.add((x, y))

    #         for position in potential_positions:
    #             for piece_index in self.pieces.get_available_pieces(self.player):
    #                 for rotation in range(unique_rotations[piece_index]):  # Use pre-computed rotations
    #                     if self.is_legal_action((*position, piece_index, rotation)):
    #                         legal_actions.append((*position, piece_index, rotation))

    #     if not legal_actions:  # If legal_actions list is empty, then it's a PASS_MOVE
    #         return [PASS_MOVE]

    #     return legal_actions

    # def is_legal_action(self, action: Tuple[int, int, int, int]) -> bool:

    #     x, y, piece_index, rotation = action
    #     piece = np.rot90(pieces[piece_index], rotation)
    #     piece_width, piece_height = piece.shape
    #     # print(f"Action: {action}")
    #     # Check if piece can be physically placed, considering its shape
    #     try:
    #         board_segment = self.board[x : x + piece_width, y : y + piece_height]

    #         if board_segment.shape != piece.shape:
    #             return False

    #     except IndexError:
    #         # Piece is off the board
    #         return False

    #     piece_mask = piece.astype(bool)
    #     board_mask = board_segment.astype(bool)

    #     # Verify if it's the first piece and if it's on the corner
    #     if self.n_turns <= 1 and self.is_on_corner(x, y, piece):
    #         if piece_mask.shape == board_mask.shape and np.any(np.logical_and(piece_mask, board_mask)):
    #             # Piece overlaps with an existing piece
    #             # print("Piece overlaps with an existing piece")
    #             return False
    #         else:
    #             return True
    #     elif self.n_turns <= 1:
    #         # print("Turn number is less or equal to 1, but the piece is not on the corner")
    #         return False
    #     else:
    #         return self.is_valid_placement(x, y, piece)

    # def is_valid_placement(self, x: int, y: int, piece: np.array) -> bool:
    #     piece_mask = piece.astype(bool)
    #     corner_touch = False

    #     piece_height, piece_width = piece.shape
    #     for i in range(piece_height):
    #         for j in range(piece_width):
    #             if piece_mask[i, j]:  # For each cell that is part of the piece
    #                 board_x = x + i
    #                 board_y = y + j

    #                 # Overlapping thing
    #                 if self.board[board_x, board_y] != 0:
    #                     return False

    #                 # Check side cells
    #                 for dx, dy in SIDES:
    #                     side_x = board_x + dx
    #                     side_y = board_y + dy
    #                     if self.is_inside(side_x, side_y) and self.board[side_x, side_y] == self.player:
    #                         return False

    #                 # Check corner cells
    #                 for dx, dy in CORNERS:
    #                     corner_x = board_x + dx
    #                     corner_y = board_y + dy
    #                     if self.is_inside(corner_x, corner_y):
    #                         if self.board[corner_x, corner_y] == self.player:
    #                             corner_touch = True

    #     # If no corner is touching the own cells, return False
    #     if not corner_touch:
    #         return False

    #     return True

    # def is_valid_placement(self, x: int, y: int, piece_mask: np.array) -> bool:
    #     board_segment = self.board[x : x + piece_mask.shape[0], y : y + piece_mask.shape[1]]

    #     # Overlapping thing
    #     if np.any(np.logical_and(board_segment, piece_mask)):
    #         return False

    #     # Check side cells
    #     for dx, dy in SIDES:
    #         side_x = x + dx
    #         side_y = y + dy
    #         if (
    #             0 <= side_x < BOARD_SIZE
    #             and 0 <= side_y < BOARD_SIZE
    #             and self.board[side_x, side_y] == self.player
    #         ):
    #             return False

    #     # Check corner cells
    #     corner_touch = any(
    #         self.board[x + dx, y + dy] == self.player
    #         for dx, dy in CORNERS
    #         if 0 <= (x + dx) < BOARD_SIZE and 0 <= (y + dy) < BOARD_SIZE
    #     )

    #     # If no corner is touching the own cells, return False
    #     if not corner_touch:
    #         return False

    #     return True

    def is_on_corner(self, x: int, y: int, piece: np.array) -> bool:
        piece_width, piece_height = piece.shape
        # print(f"Piece dimensions: {piece_width}x{piece_height}")
        # print(f"Checking if piece is on corner at position ({x}, {y})")

        # Check if any part of the piece coincides with any of the board's corners
        for i in range(piece_width):
            for j in range(piece_height):
                if piece[i, j]:  # If this part of the piece array contains the piece
                    piece_position = (x + i, y + j)
                    # print(f"Checking piece part at ({piece_position[0]}, {piece_position[1]})")
                    if piece_position in START_POSITIONS:
                        # print("Piece is on a starting corner")
                        return True

        # print("Piece is not on a starting corner")
        return False

    def evaluate_move(self, move):
        # Prefer placing larger pieces first
        _, _, piece_index, _ = move
        return np.sum(
            self.pieces.pieces[piece_index]
        )  # This works because piece cells are represented by 1's

    def is_terminal(self) -> bool:
        # If both players have passed, then we stop the game.
        if self.passes >= 2:
            return True
        # Or if the board is full or
        if np.all(self.board != 0):
            return True

        # If all players have placed all their pieces.
        if all(self.pieces.pieces_left_for_player(player + 1) == 0 for player in range(2)):
            return True

        return False

    def get_reward(self, player) -> float:
        if not self.is_terminal():
            return 0

        player_pieces_left = self.pieces.pieces_left_for_player(player)
        opponent_pieces_left = self.pieces.pieces_left_for_player(3 - player)

        if opponent_pieces_left > player_pieces_left:
            return win
        elif opponent_pieces_left < player_pieces_left:
            return loss
        else:
            return draw

    def is_capture(self, move):
        # Not applicable for this game as there are no captures
        return False

    def visualize(self) -> str:
        # player 1 represented by 1 (green color), player 2 represented by 2 (red color)
        # player 1's perimeter represented by ○ (green color), player 2's perimeter represented by ○ (red color)
        visual_board = []

        # Add a border to the top
        border_row = ["#" for _ in range(len(self.board[0]) + 2)]
        visual_board.append(border_row)

        for i, row in enumerate(self.board):
            visual_row = ["#"]  # Add border to the left
            for j, cell in enumerate(row):
                if int(cell) == 1:
                    visual_row.append(colored("■", "green"))
                elif int(cell) == 2:
                    visual_row.append(colored("▦", "red"))
                elif (i, j) in self.perimeters[1]:
                    visual_row.append(colored("○", "green"))
                elif (i, j) in self.perimeters[2]:
                    visual_row.append(colored("○", "red"))
                else:
                    visual_row.append(" ")
            visual_row.append("#")  # Add border to the right
            visual_board.append(visual_row)

        # Add a border to the bottom
        visual_board.append(border_row)

        return "\n".join(" ".join(cell for cell in row) for row in visual_board)

    def __repr__(self) -> str:
        return "blokus"

    @property
    def transposition_table_size(self):
        return 2 ** (BOARD_SIZE)


def evaluate_blokus(game_state, m_piece_diff=0.5, m_corn_diff=0.5):
    player = game_state.player
    opponent = 3 - player

    # The available pieces for current player and the opponent.
    player_pieces = game_state.pieces.pieces_left_for_player(player)
    opponent_pieces = game_state.pieces.pieces_left_for_player(opponent)

    # The available corners for current player and the opponent.
    player_corners = game_state.count_corners(player)
    opponent_corners = game_state.count_corners(opponent)

    # The difference in pieces and corners can be used as a simple evaluation.
    piece_diff = player_pieces - opponent_pieces
    corner_diff = player_corners - opponent_corners

    # A simple linear combination can be a good evaluation.
    score = m_piece_diff * piece_diff + m_corn_diff * corner_diff

    return score


def count_corners(game_state, player: int) -> int:
    corner_count = 0
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if game_state.touches_own_corner(x, y, np.array([[player]])):
                corner_count += 1
    return corner_count


# TODO Dit moet je nog verbeteren
def evaluate_blokus_enhanced(
    game_state: BlokusGameState,
    player: int,
    m_size_weight=1.0,
    m_corner_weight=1.0,
    m_turn_discount=0.9,
    m_territory_weight=1.0,
    m_piece_weight=1.0,
):
    board = game_state.board  # board is a numpy array
    opponent = 3 - player  # Assuming player numbers are 1 and 2

    piece_size = game_state.pieces.pieces_left_for_player(
        game_state.player_turn
    ) - game_state.pieces.pieces_left_for_player(opponent)

    # TODO This is wrong. It should count the number of corners, i.e. legal places where the player can play a piece
    corner_score = np.sum([CENTER_WEIGHTS[c] if board[c] == 0 else 0 for c in BOARD_CORNERS])

    # For territory control, we use convolution to calculate the number of neighbour cells for each cell
    from scipy.signal import convolve2d

    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    neighbour_counts = convolve2d(board == PLAYER_BOARD[player - 1], kernel, mode="same")
    territory_score = np.sum(CENTER_WEIGHTS[neighbour_counts == 4])

    remaining_pieces_sizes = [np.sum(piece) for piece in game_state.pieces.get_available_pieces(player)]
    opponent_pieces_sizes = [np.sum(piece) for piece in game_state.pieces.get_available_pieces(opponent)]
    remaining_pieces_score = (MAX_PIECE_COUNT - np.sum(remaining_pieces_sizes)) - (
        MAX_PIECE_COUNT - np.sum(opponent_pieces_sizes)
    )

    # Calculate the total score
    score = (
        m_size_weight * piece_size
        + m_corner_weight * corner_score
        + m_territory_weight * territory_score
        + m_piece_weight * remaining_pieces_score
    )

    # Apply the turn discount if it's not my turn
    if game_state.player != player:
        score *= m_turn_discount

    return score

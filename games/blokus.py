# cython: language_level=3

import copy
import random
from typing import Generator, List, Tuple

import cython
from cython.cimports import numpy as cnp

cnp.import_array()
import numpy as np
from termcolor import colored

from games.gamestate import GameState, draw, loss, normalize, win

if cython.compiled:
    print("Blokus is compiled.")
else:
    print("Blokus is just a lowly interpreted script.")

PLAYER_COUNT: cython.int = 2
BOARD_SIZE: cython.int = 20

PASS_MOVE: cython.tuple = (-1, -1, -1, -1)

# Define offsets for orthogonally adjacent cells (up, down, left, right)
ORTHOGONAL_NEIGHBORS: cython.list = [(0, 1), (1, 0), (0, -1), (-1, 0)]
CORNERS: cython.list = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

# We create a constant player board as it's just used for comparing against the actual board
PLAYER_BOARD: cnp.ndarray = np.array(
    [np.full((BOARD_SIZE, BOARD_SIZE), 1), np.full((BOARD_SIZE, BOARD_SIZE), 2)]
)
NEIGHBORS: cython.tuple = (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)

BOARD_CORNERS: cython.list = [
    (0, 0),
    (0, BOARD_SIZE - 1),
    (BOARD_SIZE - 1, 0),
    (BOARD_SIZE - 1, BOARD_SIZE - 1),
]


pieces: cython.list = [
    np.array([[1]], dtype=np.int32),  # monomino
    # --
    np.array([[1, 1]], dtype=np.int32),  # domino
    # --
    np.array([[1, 1, 1]], dtype=np.int32),  # tromino_I
    np.array([[1, 0], [1, 1]], dtype=np.int32),  # tromino_L
    # --
    np.array([[1, 1, 1, 1]], dtype=np.int32),  # tetromino_I
    np.array([[1, 0], [1, 0], [1, 1]], dtype=np.int32),  # tetromino_L
    np.array([[1, 1, 1], [0, 1, 0]], dtype=np.int32),  # tetromino_T
    np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int32),  # tetromino_S
    np.array([[1, 1], [1, 1]], dtype=np.int32),  # tetromino_O
    # --
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.int32),  # pentomino_V
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=np.int32),  # pentomino_F
    np.array([[1, 1, 1, 1, 1]]),  # pentomino_I
    np.array([[1, 0], [1, 0], [1, 0], [1, 1]], dtype=np.int32),  # pentomino_L
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=np.int32),  # pentomino_N
    np.array([[1, 1], [1, 1], [1, 0]], dtype=np.int32),  # pentomino_P
    np.array([[1, 1], [1, 0], [1, 1]], dtype=np.int32),  # pentomino_U
    np.array([[1, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int32),  # pentomino_J
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=np.int32),  # pentomino_T
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=np.int32),  # pentomino_W
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32),  # pentomino_X
    np.array([[1, 0], [1, 1], [1, 0], [1, 0]], dtype=np.int32),  # pentomino_Y
]
# Calculate the total size of the pieces
total_ones: cython.int = 0
# Pre-compute the number of 1's in each piece
piece_sizes: cython.dict = {}
piece_index: cython.int
piece: cnp.ndarray

for piece_index, piece in enumerate(pieces):
    piece_sizes[piece_index] = np.sum(piece)
    total_ones += np.sum(piece)


unique_rotations: cython.dict = {}
rotations: cython.list
flips: cython.list
unique: cython.list
rotated: cnp.ndarray
u: cnp.ndarray
for piece_index, piece in enumerate(pieces):
    rotations = [np.rot90(piece, rotation) for rotation in range(4)]
    # Add flipped versions of the piece and its rotations.
    flips = [np.flip(piece), np.flip(piece, 0), np.flip(piece, 1)]
    rotations += flips

    unique = [piece.tolist()]  # initialize with original piece

    for rotated in rotations:
        if not any(np.array_equal(rotated, np.array(u)) for u in unique):
            unique.append(rotated)

    unique_rotations[piece_index] = len(unique)

# Pre-compute masks for all rotated and flipped pieces
piece_masks: cython.dict = {}
rotated_pieces: cython.dict = {}

for piece_index, piece in enumerate(pieces):
    rotations = [np.rot90(piece, rotation) for rotation in range(unique_rotations[piece_index])]
    # Add flipped versions of the piece and its rotations.
    flips = [np.flip(piece), np.flip(piece, 0), np.flip(piece, 1)]
    rotations += flips

    rotated_pieces[piece_index] = [np.array(rotation).astype(np.int32) for rotation in rotations]
    piece_masks[piece_index] = [rotation.astype(bool) for rotation in rotated_pieces[piece_index]]

MAX_PIECE_COUNT: cython.int = total_ones * 2


@cython.cclass
class BlokusPieces:
    rem_pieces = cython.declare(cython.dict)
    pieces_count = cython.declare(cython.dict)
    pieces_size = cython.declare(cython.dict)

    def __init__(self, init_state=False):
        if init_state:
            self.rem_pieces = {(color, i): True for color in [1, 2, 3, 4] for i in range(len(pieces))}
            self.pieces_count = {i: len(pieces) for i in [1, 2, 3, 4]}
            self.pieces_size = {
                i: sum([np.sum(pieces[piece_i]) for piece_i in range(len(pieces))]) for i in [1, 2, 3, 4]
            }

    def copy(self):
        return copy.deepcopy(self)

    @cython.ccall
    @cython.locals(piece_index=cython.int, color=cython.int, key=cython.tuple)
    def play_piece(self, piece_index, color):
        key = (color, piece_index)
        if self.rem_pieces[key]:
            self.rem_pieces[key] = False
            self.pieces_count[color] -= 1
            self.pieces_size[color] -= piece_sizes[piece_index]
        else:
            raise ValueError(f"Player {color} doesn't have piece {piece_index} available.")

    @cython.ccall
    @cython.locals(piece_index=cython.int, color=cython.int, color_i=cython.int, piece_i=cython.int)
    def avail_pieces_for_color(self, color) -> cython.list:
        return [
            piece_i for (color_i, piece_i), is_av in self.rem_pieces.items() if is_av and color_i == color
        ]

    @cython.ccall
    @cython.locals(color=cython.int)
    def pieces_left_for_color(self, color) -> cython.int:
        return self.pieces_count[color]

    @cython.ccall
    @cython.locals(player=cython.int)
    def pieces_left_for_player(self, player) -> cython.int:
        if player == 1:
            return self.pieces_count[1] + self.pieces_count[3]
        else:
            return self.pieces_count[2] + self.pieces_count[4]

    @cython.ccall
    @cython.locals(player=cython.int)
    def sum_piece_size(self, player) -> cython.int:
        if player == 1:
            return self.pieces_size[1] + self.pieces_count[3]
        else:
            return self.pieces_size[2] + self.pieces_count[4]

    def __str__(self):
        return f"BlokusPieces:\n Pieces count: {self.pieces_count}"


class BlokusGameState:
    # Changed zobrist_table size to include 4 players
    zobrist_table = np.random.randint(
        low=0,
        high=np.iinfo(np.uint32).max,
        size=(BOARD_SIZE, BOARD_SIZE, 5),  # 4 players + 1 for empty state
        dtype=np.uint32,
    )

    def __init__(self, board=None, pieces=None, player=1, n_turns=0, passed=None):
        if passed is None:
            passed = {1: False, 2: False, 3: False, 4: False}

        if pieces is None:
            self.pieces = BlokusPieces(init_state=True)
        else:
            self.pieces = pieces

        self.player = player

        if board is not None:
            self.board = board
        else:
            self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
            self.board_hash = self._calculate_board_hash()

        self.n_turns = n_turns
        self.color = (n_turns % 4) + 1
        self.passed = passed
        self.positions_checked = 0

    def _calculate_board_hash(self):
        board_hash = np.uint32(0)
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                player = int(self.board[x, y])
                board_hash ^= self.zobrist_table[x, y, player]
        return board_hash

    def skip_turn(self):
        """Used for the null-move heuristic in alpha-beta search"""
        new_state = BlokusGameState(
            board=np.copy(self.board),
            pieces=self.pieces.copy(),
            player=3 - self.player,
            n_turns=self.n_turns + 1,
            passed=self.passed.copy(),
        )
        new_state.board_hash = new_state._calculate_board_hash()
        return new_state

    def apply_action(self, action: Tuple[int, int, int, int]) -> BlokusGameState:
        new_state = BlokusGameState(
            board=np.copy(self.board),
            pieces=self.pieces.copy(),
            player=self.player,
            n_turns=self.n_turns,
            passed=self.passed.copy(),
        )

        if action == PASS_MOVE:
            new_state.passed[self.color] = True
        else:
            x, y, piece_index, rotation = action
            update_state(new_state.board, x, y, piece_index, rotation, self.color)
            new_state.pieces.play_piece(piece_index, self.color)

        new_state.n_turns += 1
        new_state.color = (new_state.n_turns % 4) + 1
        new_state.player = 3 - self.player
        new_state.board_hash = new_state._calculate_board_hash()

        return new_state

    def get_random_action(self):
        if self.passed[self.color]:
            return PASS_MOVE
        # Store already checked actions to prevent repeating them
        checked_actions = set()

        perimeter_points = list(self.perimeters[self.color])
        avail_pieces = self.pieces.avail_pieces_for_color(self.color)
        total_possible_actions = sum(unique_rotations[piece_index] for piece_index in avail_pieces) * len(
            perimeter_points
        )

        while len(checked_actions) < total_possible_actions:
            # Select a random perimeter point
            if perimeter_points:
                px, py = random.choice(perimeter_points)
            else:  # No available perimeter points
                return [PASS_MOVE]

            # Select a random piece
            if avail_pieces:
                piece_index = random.choice(avail_pieces)
            else:  # No available pieces
                return [PASS_MOVE]

            rotation = random.randint(0, unique_rotations[piece_index] - 1)
            piece = rotated_pieces[piece_index][rotation]
            piece_width, piece_height = piece.shape

            dx = random.randint(max(-piece_width + 1, -px), min(piece_width, BOARD_SIZE - px))
            dy = random.randint(max(-piece_height + 1, -py), min(piece_height, BOARD_SIZE - py))

            x, y = px + dx, py + dy

            # Check if piece can be physically placed, considering its shape
            if not (0 <= x + piece_width - 1 < BOARD_SIZE and 0 <= y + piece_height - 1 < BOARD_SIZE):
                continue

            action = (x, y, piece_index, rotation)
            if action in checked_actions:
                continue

            checked_actions.add(action)
            if self.is_legal_action(x, y, piece, piece_masks[piece_index][rotation]):
                return action

        # All possible actions have been tried and are illegal, return a PASS_MOVE
        return PASS_MOVE

    def yield_legal_actions(self) -> Generator[Tuple[int, int, int, int], None, None]:
        if self.passed[self.color]:
            yield PASS_MOVE
            return

        self.positions_checked = 0
        checked_actions = set()
        perimeter_points = list(self.perimeters[self.color])
        random.shuffle(perimeter_points)  # Shuffle perimeter points

        avail_pieces = self.pieces.avail_pieces_for_color(self.color)
        random.shuffle(avail_pieces)  # Shuffle available pieces

        for px, py in perimeter_points:
            # The first 4 moves must be played in the corners
            if self.n_turns <= 3 and (px, py) not in BOARD_CORNERS:
                continue

            for piece_index in avail_pieces:
                rotations = list(range(unique_rotations[piece_index]))  # Create list of rotations
                random.shuffle(rotations)  # Shuffle rotations
                for rotation in rotations:  # Use pre-computed rotations
                    piece = rotated_pieces[piece_index][rotation]
                    piece_width, piece_height = piece.shape

                    dx_values = list(range(max(-piece_width + 1, -px), min(piece_width, BOARD_SIZE - px)))
                    random.shuffle(dx_values)  # Shuffle dx_values
                    for dx in dx_values:
                        dy_values = list(
                            range(max(-piece_height + 1, -py), min(piece_height, BOARD_SIZE - py))
                        )
                        random.shuffle(dy_values)  # Shuffle dy_values
                        for dy in dy_values:
                            x, y = px + dx, py + dy
                            # Check if piece can be physically placed, considering its shape
                            if not (
                                0 <= x + piece_width - 1 < BOARD_SIZE
                                and 0 <= y + piece_height - 1 < BOARD_SIZE
                            ):
                                continue

                            action = (x, y, piece_index, rotation)
                            if action in checked_actions:
                                continue

                            checked_actions.add(action)
                            if self.is_legal_action(x, y, piece, piece_masks[piece_index][rotation]):
                                yield action

        # If we haven't yielded any actions, then it's a PASS_MOVE
        yield PASS_MOVE

    def get_legal_actions(self) -> list:
        if self.passed[self.color]:
            return [PASS_MOVE]

        return get_legal_actions(
            self.board, self.color, self.n_turns, self.pieces.avail_pieces_for_color(self.color)
        )

    def evaluate_moves(self, moves):
        """
        Evaluates a list of moves, preferring the placement of larger pieces first.

        :param moves: The list of moves to evaluate.
        :return: The list of scores for the moves.
        """
        scores = []
        for move in moves:
            _, _, piece_index, _ = move
            if piece_index != -1:  # In the case of a pass move
                # This works because piece cells are represented by 1's
                scores.append((move, piece_sizes[piece_index]))
            else:
                scores.append((move, 0))
        return scores

    def evaluate_move(self, move):
        # Prefer placing larger pieces first
        _, _, piece_index, _ = move
        if piece_index != -1:  # In the case of a pass move
            return piece_sizes[piece_index]  # This works because piece cells are represented by 1's
        return 0

    def is_terminal(self) -> bool:
        # If both players have passed, then we stop the game.
        if all((self.passed[1], self.passed[2], self.passed[3], self.passed[4])):
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

        player_pieces_left = self.pieces.sum_piece_size(player)
        opponent_pieces_left = self.pieces.sum_piece_size(3 - player)

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
        # Adding more colors for additional players
        colors = ["green", "red", "blue", "yellow"]
        visual_board = []

        perimeters = {}
        for i in range(4):
            perimeters[i] = find_corners_for_color(self.board, i + 1)

        # Add a border to the top
        border_row = ["#" for _ in range(len(self.board[0]) + 2)]
        visual_board.append(border_row)

        for i, row in enumerate(self.board):
            visual_row = ["#"]  # Add border to the left
            for j, cell in enumerate(row):
                if int(cell) > 0:
                    visual_row.append(colored("■", colors[int(cell) - 1]))
                elif (i, j) in perimeters[0]:
                    visual_row.append(colored("○", colors[0]))
                elif (i, j) in perimeters[1]:
                    visual_row.append(colored("○", colors[1]))
                elif (i, j) in perimeters[2]:
                    visual_row.append(colored("○", colors[2]))
                elif (i, j) in perimeters[3]:
                    visual_row.append(colored("○", colors[3]))
                else:
                    visual_row.append(" ")

            visual_row.append("#")  # Add border to the right
            visual_board.append(visual_row)

        # Add a border to the bottom
        visual_board.append(border_row)

        return "\n".join(" ".join(cell for cell in row) for row in visual_board) + "\n"

    def __repr__(self) -> str:
        return "blokus"

    @property
    def transposition_table_size(self):
        return 2 ** (BOARD_SIZE)


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def update_state(
    board: cython.int[:, :],
    x: cython.int,
    y: cython.int,
    piece_index: cython.int,
    rotation: cython.int,
    color: cython.int,
):
    piece: cython.int[:, :] = rotated_pieces[piece_index][rotation]
    indices: cython.int[:, :] = np.argwhere(piece).astype(np.int32)
    i: cython.int
    dx: cython.int
    dy: cython.int

    for i in range(indices.shape[0]):
        dx = indices[i, 0]
        dy = indices[i, 1]
        board[x + dx, y + dy] = color


@cython.ccall
@cython.locals(
    px=cython.int,
    py=cython.int,
    piece=cython.int[:, :],
    piece_index=cython.int,
    rotation=cython.int,
    piece_width=cython.int,
    piece_height=cython.int,
    dx=cython.int,
    dy=cython.int,
    x=cython.int,
    y=cython.int,
    p=cython.int,
)
def get_legal_actions(
    board: cython.int[:, :],
    color: cython.int,
    n_turns: cython.int,
    avail_pieces: cython.list,
) -> cython.list:
    legal_actions: cython.list = []
    checked_actions: cython.set = set()
    # Get a set of points around which to look, preventing us from checking the entire board for each piece
    if n_turns > 3:
        perimeter_points = find_corners_for_color(board, color)
    else:
        perimeter_points = BOARD_CORNERS

    for px, py in perimeter_points:
        for p in range(len(avail_pieces)):
            piece_index = avail_pieces[p]
            for rotation in range(unique_rotations[piece_index]):  # Use pre-computed rotations
                piece = rotated_pieces[piece_index][rotation]
                piece_width = piece.shape[0]
                piece_height = piece.shape[1]

                for dx in range(max(-piece_width + 1, -px), min(piece_width, 20 - px)):
                    for dy in range(max(-piece_height + 1, -py), min(piece_height, 20 - py)):
                        x, y = px + dx, py + dy
                        # Check if piece can be physically placed, considering its shape
                        if not (0 <= x + piece_width - 1 < 20 and 0 <= y + piece_height - 1 < 20):
                            continue

                        action = (x, y, piece_index, rotation)

                        if action in checked_actions:
                            continue
                        checked_actions.add(action)

                        if is_legal_action(x, y, piece, n_turns, color, board):
                            legal_actions.append(action)

    if not legal_actions:  # If legal_actions list is empty, then it's a PASS_MOVE
        return [PASS_MOVE]

    return legal_actions


@cython.ccall
@cython.locals(
    x=cython.int,
    y=cython.int,
    i=cython.int,
    j=cython.int,
    dx=cython.int,
    dy=cython.int,
    corner_touch=cython.bint,
    n_turns=cython.int,
    color=cython.int,
    board=cython.int[:, :],
    piece=cython.int[:, :],
    new_x=cython.int,
    new_y=cython.int,
)
def is_legal_action(
    x: cython.int,
    y: cython.int,
    piece: cython.int[:, :],
    n_turns: cython.int,
    color: cython.int,
    board: cython.int[:, :],
) -> cython.bint:
    # Verify if it's the first piece and if it's on the corner
    if n_turns <= 3 and is_on_board_corner(x, y, piece):
        return no_overlap(piece, board, x, y)
    elif n_turns <= 3:
        return False

    # Piece should not overlap with an existing piece or opponent's piece
    if not no_overlap(piece, board, x, y):
        return False

    corner_touch: cython.bint = 0
    indices: cython.int[:, :] = np.argwhere(piece).astype(np.int32)

    for k in range(indices.shape[0]):
        dx = indices[k, 0]
        dy = indices[k, 1]
        # We only have to find one point where the piece touches a corner
        if not corner_touch:
            # Check diagonals
            for i in range(-1, 2, 2):
                for j in range(-1, 2, 2):
                    # If the new position is within the board
                    if 0 <= x + dx + i < board.shape[0] and 0 <= y + dy + j < board.shape[1]:
                        # If there's a piece on the diagonal
                        if board[x + dx + i, y + dy + j] == color:
                            corner_touch = 1
                            break
                if corner_touch:
                    break

        # Check orthogonals
        for i in range(-1, 2):
            for j in range(-1, 2):
                if abs(i) != abs(j) and (
                    0 <= x + dx + i < board.shape[0]
                    and 0 <= y + dy + j < board.shape[1]
                    and board[x + dx + i, y + dy + j] == color
                ):
                    return False  # Some cell of the piece is orthogonally touching a cell of an existing piece of the player

    return corner_touch


@cython.cfunc
@cython.locals(
    piece=cython.int[:, :], board=cython.int[:, :], x=cython.int, y=cython.int, i=cython.int, j=cython.int
)
def no_overlap(piece, board, x, y) -> cython.bint:
    for i in range(piece.shape[0]):
        for j in range(piece.shape[1]):
            if piece[i, j] != 0 and board[x + i, y + j] != 0:  # Equivalent of np.logical_and
                return 0  # Found overlap, so return False
    return 1  # No overlap was found, so return True


@cython.cfunc
@cython.locals(
    piece_x=cython.int,
    piece_y=cython.int,
    i=cython.int,
    j=cython.int,
    piece_width=cython.int,
    piece_height=cython.int,
)
def is_on_board_corner(x: cython.int, y: cython.int, piece: cython.int[:, :]) -> cython.bint:
    piece_width = piece.shape[0]
    piece_height = piece.shape[1]
    # Check if any part of the piece coincides with any of the board's corners
    for i in range(piece_width):
        for j in range(piece_height):
            if piece[i, j]:  # If this part of the piece array contains the piece
                piece_x = x + i
                piece_y = y + j
                # Check if the piece is at any of the board's corners
                if (piece_x == 0 or piece_x == 19) and (piece_y == 0 or piece_y == 19):
                    return True
    return False


@cython.ccall
@cython.locals(
    board=cython.int[:, :],
    color=cython.int,
    i=cython.int,
    j=cython.int,
    dx=cython.int,
    dy=cython.int,
    new_i=cython.int,
    new_j=cython.int,
    orthogonal_touch=cython.bint,
    ortho_i=cython.int,
    ortho_j=cython.int,
)
def find_corners_for_color(board, color) -> cython.set:
    corner_points = set()

    # Iterate over the board
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            # Check if the current cell is of the given color
            if board[i, j] == color:
                # Check diagonals
                for dx in range(-1, 2, 2):
                    for dy in range(-1, 2, 2):
                        new_i, new_j = i + dx, j + dy
                        # If the diagonal cell is inside the board and is empty
                        if (
                            0 <= new_i < board.shape[0]
                            and 0 <= new_j < board.shape[1]
                            and board[new_i, new_j] == 0
                        ):
                            # Check orthogonals
                            orthogonal_touch = 0
                            for di in range(-1, 2):
                                for dj in range(-1, 2):
                                    if abs(di) != abs(dj):
                                        ortho_i, ortho_j = new_i + di, new_j + dj
                                        # If the orthogonal cell is inside the board and has the given color
                                        if (
                                            0 <= ortho_i < board.shape[0]
                                            and 0 <= ortho_j < board.shape[1]
                                            and board[ortho_i, ortho_j] == color
                                        ):
                                            orthogonal_touch = 1
                                            break

                                if orthogonal_touch:
                                    break

                            # if the point we found is not orthoginally connected, add it to the corner points
                            if not orthogonal_touch:
                                corner_points.add((new_i, new_j))

    return corner_points


@cython.ccall
@cython.locals(
    game_state=cython.object,
    m_piece_diff=cython.double,
    m_corn_diff=cython.double,
    m_piece_size=cython.double,
    m_turn=cython.double,
    a=cython.int,
    norm=cython.bint,
    player=cython.int,
    opponent=cython.int,
    p_color1=cython.int,
    p_color2=cython.int,
    o_color1=cython.int,
    o_color2=cython.int,
    player_pieces=cython.int,
    opponent_pieces=cython.int,
    player_corners=cython.int,
    opponent_corners=cython.int,
    piece_diff=cython.int,
    corner_diff=cython.int,
    player_pieces_sizes=cython.int,
    opponent_pieces_sizes=cython.int,
    piece_size_diff=cython.int,
    score=cython.double,
)
def evaluate_blokus(
    game_state,
    m_piece_diff=0.33,
    m_corn_diff=0.33,
    m_piece_size=0.33,
    m_turn=0.9,
    a=50,
    norm=0,
) -> cython.double:
    player = game_state.player
    opponent = 3 - player
    board: cython.int[:, :] = game_state.board
    state_pieces: BlokusPieces = game_state.pieces

    if player == 1:
        p_color1 = 1
        p_color2 = 3
        o_color1 = 2
        o_color2 = 4
    else:
        p_color1 = 2
        p_color2 = 4
        o_color1 = 1
        o_color2 = 3

    player_pieces = state_pieces.pieces_left_for_player(player)
    opponent_pieces = state_pieces.pieces_left_for_player(opponent)

    player_corners = len(find_corners_for_color(board, p_color1)) + len(
        find_corners_for_color(board, p_color2)
    )
    opponent_corners = len(find_corners_for_color(board, o_color1)) + len(
        find_corners_for_color(board, o_color2)
    )

    piece_diff = opponent_pieces - player_pieces
    corner_diff = player_corners - opponent_corners

    player_pieces_sizes = state_pieces.sum_piece_size(player)
    opponent_pieces_sizes = state_pieces.sum_piece_size(opponent)

    piece_size_diff = opponent_pieces_sizes - player_pieces_sizes

    score = m_piece_diff * piece_diff + m_corn_diff * corner_diff + m_piece_size * piece_size_diff

    score = score if game_state.player == player else m_turn * score

    if norm:
        return normalize(score, a)
    else:
        return score

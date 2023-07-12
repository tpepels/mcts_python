import copy
import random
from typing import Generator, List, Tuple
import numpy as np
from termcolor import colored

from games.gamestate import GameState, win, loss, draw

PLAYER_COUNT = 2
BOARD_SIZE = 20

PASS_MOVE = (-1, -1, -1, -1)

# Define offsets for orthogonally adjacent cells (up, down, left, right)
ORTHOGONAL_NEIGHBORS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
CORNERS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
# We create a constant player board as it's just used for comparing against the actual board
PLAYER_BOARD = [np.full((BOARD_SIZE, BOARD_SIZE), 1), np.full((BOARD_SIZE, BOARD_SIZE), 2)]
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
# Pre-compute the number of 1's in each piece
piece_sizes = {}
for piece_index, piece in enumerate(pieces):
    piece_sizes[piece_index] = np.sum(piece)
    total_ones += np.sum(piece)


unique_rotations = {}
for i, piece in enumerate(pieces):
    rotations = [np.rot90(piece, rotation) for rotation in range(4)]
    # Add flipped versions of the piece and its rotations.
    flips = [np.flip(piece), np.flip(piece, 0), np.flip(piece, 1)]
    rotations += flips

    unique = [piece.tolist()]  # initialize with original piece

    for rotated in rotations:
        if not any(np.array_equal(rotated, np.array(u)) for u in unique):
            unique.append(rotated.tolist())

    unique_rotations[i] = len(unique)

# Pre-compute rotated and flipped pieces
piece_masks = {}
rotated_pieces = {}
for piece_index, piece in enumerate(pieces):
    rotations = [np.rot90(piece, rotation) for rotation in range(unique_rotations[piece_index])]
    # Add flipped versions of the piece and its rotations.
    flips = [np.flip(piece), np.flip(piece, 0), np.flip(piece, 1)]
    rotations += flips

    rotated_pieces[piece_index] = rotations
    piece_masks[piece_index] = [rotation.astype(bool) for rotation in rotated_pieces[piece_index]]

MAX_PIECE_COUNT = total_ones * 2


class BlokusPieces:
    def __init__(self, init_state=False):
        if init_state:
            self.rem_pieces = {(color, i): True for color in [1, 2, 3, 4] for i in range(len(pieces))}
            self.pieces_count = {i: len(pieces) for i in [1, 2, 3, 4]}
            self.pieces_size = {
                i: sum([np.sum(pieces[piece_i]) for piece_i in range(len(pieces))]) for i in [1, 2, 3, 4]
            }

    def copy(self):
        return copy.deepcopy(self)

    def play_piece(self, piece_index, color):
        key = (color, piece_index)
        if self.rem_pieces[key]:
            self.rem_pieces[key] = False
            self.pieces_count[color] -= 1
            self.pieces_size[color] -= piece_sizes[piece_index]
        else:
            raise ValueError(f"Player {color} doesn't have piece {piece_index} available.")

    def avail_pieces_for_color(self, color):
        return [
            piece_i for (color_i, piece_i), is_av in self.rem_pieces.items() if is_av and color_i == color
        ]

    def pieces_left_for_color(self, color):
        return self.pieces_count[color]

    def pieces_left_for_player(self, player):
        if player == 1:
            return self.pieces_count[1] + self.pieces_count[3]
        else:
            return self.pieces_count[2] + self.pieces_count[4]

    def sum_piece_size(self, player):
        if player == 1:
            return self.pieces_size[1] + self.pieces_count[3]
        else:
            return self.pieces_size[2] + self.pieces_count[4]

    def __str__(self):
        return f"BlokusPieces:\n Pieces count: {self.pieces_count}"


class BlokusGameState(GameState):
    # Changed zobrist_table size to include 4 players
    zobrist_table = np.random.randint(
        low=0,
        high=np.iinfo(np.uint64).max,
        size=(BOARD_SIZE, BOARD_SIZE, 5),  # 4 players + 1 for empty state
        dtype=np.uint64,
    )

    def __init__(self, board=None, pieces=None, player=1, n_turns=0, passed=None, perimeters=None):
        if passed is None:
            passed = {1: False, 2: False, 3: False, 4: False}

        if pieces is None:
            self.pieces = BlokusPieces(init_state=True)
        else:
            self.pieces = pieces

        self.player = player

        if board is not None:
            self.board = board
            self.perimeters = perimeters
        else:
            self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
            self.board_hash = self._calculate_board_hash()
            # Initialize the perimeters as a dictionary with player numbers as keys
            self.perimeters = {
                1: set(BOARD_CORNERS.copy()),
                2: set(BOARD_CORNERS.copy()),
                3: set(BOARD_CORNERS.copy()),
                4: set(BOARD_CORNERS.copy()),
            }

        self.n_turns = n_turns
        self.color = (n_turns % 4) + 1
        self.passed = passed
        self.positions_checked = 0

    def _calculate_board_hash(self):
        board_hash = np.uint64(0)
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
            perimeters=copy.deepcopy(self.perimeters),
        )
        new_state.board_hash = new_state._calculate_board_hash()
        return new_state

    def apply_action(self, action: Tuple[int, int, int, int]) -> "BlokusGameState":
        new_state = BlokusGameState(
            board=np.copy(self.board),
            pieces=self.pieces.copy(),
            player=self.player,
            n_turns=self.n_turns,
            passed=self.passed.copy(),
            perimeters=copy.deepcopy(self.perimeters),
        )

        if action == PASS_MOVE:
            new_state.passed[self.color] = True
        else:
            x, y, piece_index, rotation = action
            piece = rotated_pieces[piece_index][rotation]
            piece_width, piece_height = piece.shape
            player_piece = np.where(piece, self.color, 0)

            board_mask = new_state.board[x : x + piece_width, y : y + piece_height].astype(bool)
            update_mask = np.logical_or(board_mask, piece)

            updated_board_segment = np.where(
                update_mask, player_piece, new_state.board[x : x + piece_width, y : y + piece_height]
            )

            new_state.board[x : x + piece_width, y : y + piece_height] = updated_board_segment

            piece_mask = piece_masks[piece_index][rotation]
            idxs = np.nonzero(piece_mask)

            new_state.perimeters[self.color] -= set((x + dx, y + dy) for dx, dy in zip(*idxs))

            orthogonal_cells = set(
                (x + dx + ddx, y + dy + ddy) for dx, dy in zip(*idxs) for ddx, ddy in ORTHOGONAL_NEIGHBORS
            )
            orthogonal_cells = set(
                (dx, dy) for dx, dy in orthogonal_cells if 0 <= dx < BOARD_SIZE and 0 <= dy < BOARD_SIZE
            )

            new_state.perimeters[self.color] -= orthogonal_cells
            for color in new_state.perimeters:
                if color != self.color:
                    new_state.perimeters[color] -= orthogonal_cells
                    new_state.perimeters[color] -= set((x + dx, y + dy) for dx, dy in zip(*idxs))

            neighboring_cells = set(
                (x + dx + ddx, y + dy + ddy) for dx, dy in zip(*idxs) for ddx, ddy in NEIGHBORS
            )
            neighboring_cells = set(
                (dx, dy) for dx, dy in neighboring_cells if 0 <= dx < BOARD_SIZE and 0 <= dy < BOARD_SIZE
            )
            neighboring_cells = set(
                (dx, dy)
                for dx, dy in neighboring_cells
                if new_state.board[dx, dy] == 0
                and all(
                    new_state.board[dx + ddx, dy + ddy] == 0
                    for ddx, ddy in ORTHOGONAL_NEIGHBORS
                    if 0 <= dx + ddx < BOARD_SIZE and 0 <= dy + ddy < BOARD_SIZE
                )
            )

            new_state.perimeters[self.color].update(neighboring_cells)
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

    import random

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

    def get_legal_actions(self) -> List[Tuple[int, int, int, int]]:
        if self.passed[self.color]:
            return [PASS_MOVE]

        self.positions_checked = 0
        legal_actions = []
        checked_actions = set()
        perimeter_points = set(self.perimeters[self.color])

        avail_pieces = self.pieces.avail_pieces_for_color(self.color)

        for px, py in perimeter_points:
            # The first 4 moves must be played in the corners
            if self.n_turns <= 3 and (px, py) not in BOARD_CORNERS:
                continue

            for piece_index in avail_pieces:
                for rotation in range(unique_rotations[piece_index]):  # Use pre-computed rotations
                    piece = rotated_pieces[piece_index][rotation]
                    piece_width, piece_height = piece.shape

                    for dx in range(max(-piece_width + 1, -px), min(piece_width, BOARD_SIZE - px)):
                        for dy in range(max(-piece_height + 1, -py), min(piece_height, BOARD_SIZE - py)):
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
                                legal_actions.append(action)

        if not legal_actions:  # If legal_actions list is empty, then it's a PASS_MOVE
            return [PASS_MOVE]

        return legal_actions

    def is_legal_action(self, x, y, piece, piece_mask) -> bool:
        self.positions_checked += 1
        piece_width, piece_height = piece.shape

        board_mask = self.board[x : x + piece_width, y : y + piece_height].astype(bool)

        # Verify if it's the first piece and if it's on the corner
        if self.n_turns <= 3 and self.is_on_board_corner(x, y, piece):
            return not np.any(
                np.logical_and(piece_mask, board_mask)
            )  # Piece should not overlap with an existing piece

        elif self.n_turns <= 3:
            return False

        # Piece should not overlap with an existing piece or opponent's piece
        if np.any(np.logical_and(piece_mask, board_mask)):
            return False

        corner_touch = False
        for dx, dy in np.argwhere(piece_mask):
            # Only check a corner touch once
            if not corner_touch and ((x + dx, y + dy) in self.perimeters[self.color]):
                corner_touch = True  # At least one cell of the piece is on the player's perimeter

            for dx2, dy2 in ORTHOGONAL_NEIGHBORS:
                if (
                    0 <= x + dx + dx2 < BOARD_SIZE
                    and 0 <= y + dy + dy2 < BOARD_SIZE
                    and self.board[x + dx + dx2, y + dy + dy2] == self.color
                ):
                    return False  # Some cell of the piece is orthogonally touching a cell of an existing piece of the player

        return corner_touch

    def is_on_board_corner(self, x: int, y: int, piece: np.array) -> bool:
        piece_width, piece_height = piece.shape
        # Check if any part of the piece coincides with any of the board's corners
        for i in range(piece_width):
            for j in range(piece_height):
                if piece[i, j]:  # If this part of the piece array contains the piece
                    piece_position = (x + i, y + j)
                    if piece_position in BOARD_CORNERS:
                        return True
        return False

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

        # Add a border to the top
        border_row = ["#" for _ in range(len(self.board[0]) + 2)]
        visual_board.append(border_row)

        for i, row in enumerate(self.board):
            visual_row = ["#"]  # Add border to the left
            for j, cell in enumerate(row):
                if int(cell) > 0:
                    visual_row.append(colored("■", colors[int(cell) - 1]))
                elif (i, j) in self.perimeters[1]:
                    visual_row.append(colored("○", colors[0]))
                elif (i, j) in self.perimeters[2]:
                    visual_row.append(colored("○", colors[1]))
                elif (i, j) in self.perimeters[3]:
                    visual_row.append(colored("○", colors[2]))
                elif (i, j) in self.perimeters[4]:
                    visual_row.append(colored("○", colors[3]))
                else:
                    visual_row.append(" ")

            visual_row.append("#")  # Add border to the right
            visual_board.append(visual_row)

        # Add a border to the bottom
        visual_board.append(border_row)

        return "\n".join(" ".join(cell for cell in row) for row in visual_board)

    def check_perimeter_overlap(self, i: int, j: int) -> bool:
        # Check if there is a piece on the board at the given coordinates
        if self.board[i][j] == 0:
            return False

        # Check if the coordinates are in the perimeter of any player
        for perimeter in self.perimeters.values():
            if (i, j) in perimeter:
                return True

        return False

    def __repr__(self) -> str:
        return "blokus"

    @property
    def transposition_table_size(self):
        return 2 ** (BOARD_SIZE)


def evaluate_blokus(
    game_state: BlokusGameState, m_piece_diff=0.33, m_corn_diff=0.33, m_piece_size=0.33, m_turn=0.9
):
    player = game_state.player
    opponent = 3 - player

    p_colors = (1, 3) if player == 1 else (2, 4)
    o_colors = (2, 4) if player == 1 else (1, 3)
    # The available pieces for current player and the opponent.
    player_pieces = game_state.pieces.pieces_left_for_player(player)
    opponent_pieces = game_state.pieces.pieces_left_for_player(opponent)

    # print(f"Player pieces left: {player_pieces}, Opponent pieces left: {opponent_pieces}")

    # The available corners for current player and the opponent.
    player_corners = len(game_state.perimeters[p_colors[0]]) + len(game_state.perimeters[p_colors[1]])
    opponent_corners = len(game_state.perimeters[o_colors[0]]) + len(game_state.perimeters[o_colors[1]])
    # print(f"Player corners available: {player_corners}, Opponent corners available: {opponent_corners}")

    # The difference in pieces and corners can be used as a simple evaluation.
    piece_diff = opponent_pieces - player_pieces  # Having less pieces is beneficial
    corner_diff = player_corners - opponent_corners  # Having more corners is desired
    # print(f"Piece difference: {piece_diff}, Corner difference: {corner_diff}")

    # This method can reference player (it already takes care of summing the colors)
    player_pieces_sizes = game_state.pieces.sum_piece_size(player)
    opponent_pieces_sizes = game_state.pieces.sum_piece_size(opponent)
    # print(f"{player_pieces_sizes=}, {opponent_pieces_sizes=}")
    piece_size_diff = opponent_pieces_sizes - player_pieces_sizes
    # print(f"Piece size difference: {piece_size_diff}")

    # A simple linear combination can be a good evaluation.
    score = m_piece_diff * piece_diff + m_corn_diff * corner_diff + m_piece_size * piece_size_diff
    # print(f"Score: {score}")

    # If I am to move I will by definition be a bit behind the opponent as they will have one more piece on the board
    return score if game_state.player == player else m_turn * score

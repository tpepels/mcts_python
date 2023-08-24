# cython: language_level=3, infer_types=True, cdivision=True, boundscheck=False, wraparound=False, nonecheck=False

import random
from typing import Generator, Tuple

import cython
from cython.cimports import numpy as cnp
from cython.cimports.includes import c_random, normalize, GameState, draw, loss, win

cnp.import_array()
import numpy as np
from termcolor import colored

if cython.compiled:
    print("Blokus is compiled.")
else:
    print("Blokus is just a lowly interpreted script.")

PLAYER_COUNT: cython.int = 2
BOARD_SIZE: cython.int = 20

PASS_MOVE: tuple[cython.int, cython.int, cython.int, cython.int] = (-1, -1, -1, -1)

# Define offsets for orthogonally adjacent cells (up, down, left, right)
ORTHOGONAL_NEIGHBORS: cython.list = [(0, 1), (1, 0), (0, -1), (-1, 0)]
CORNERS: cython.list = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

# We create a constant player board as it's just used for comparing against the actual board
PLAYER_BOARD: cnp.ndarray = np.array(
    [np.full((BOARD_SIZE, BOARD_SIZE), 1), np.full((BOARD_SIZE, BOARD_SIZE), 2)]
)


BOARD_CORNERS: cython.set = set(
    [
        (0, 0),
        (0, BOARD_SIZE - 1),
        (BOARD_SIZE - 1, 0),
        (BOARD_SIZE - 1, BOARD_SIZE - 1),
    ]
)
colors = ["green", "red", "blue", "yellow"]

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


rotated_pieces: cython.dict = {}
piece_indices: cython.dict = {}
for piece_index, piece in enumerate(pieces):
    rotations = [np.rot90(piece, rotation) for rotation in range(unique_rotations[piece_index])]
    # Add flipped versions of the piece and its rotations.
    flips = [np.flip(piece), np.flip(piece, 0), np.flip(piece, 1)]
    rotations += flips

    rotated_pieces[piece_index] = [np.array(rotation).astype(np.int32) for rotation in rotations]
    piece_indices[piece_index] = [np.argwhere(rotation).astype(np.int32) for rotation in rotations]

MAX_PIECE_COUNT: cython.int = total_ones * 2


@cython.cclass
class BlokusPieces:
    rem_pieces = cython.declare(cython.short[:, :])
    pieces_count = cython.declare(cython.short[:])
    pieces_size = cython.declare(cython.short[:])

    def __init__(self, init_state=False):
        if init_state:
            # Initialized all pieces to be available (True)
            self.rem_pieces = np.ones((4, len(pieces)), dtype=np.int16)
            # Initialized all counts to be the total number of pieces
            self.pieces_count = np.full(4, len(pieces), dtype=np.int16)
            # Initialize all sizes to the total size of pieces
            total_size = np.sum([np.sum(piece) for piece in pieces])
            self.pieces_size = np.full(4, total_size, dtype=np.int16)
        else:
            # Create uninitialized arrays for fast initialization
            self.rem_pieces = np.empty((4, len(pieces)), dtype=np.int16)
            self.pieces_count = np.empty(4, dtype=np.int16)
            self.pieces_size = np.empty(4, dtype=np.int16)

    @cython.ccall
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.locals(new_obj=BlokusPieces)
    def custom_copy(self) -> BlokusPieces:
        new_obj = BlokusPieces(init_state=True)
        # Copying rem_pieces array
        new_obj.rem_pieces[:, :] = self.rem_pieces
        # Copying pieces_count array
        new_obj.pieces_count[:] = self.pieces_count
        # Copying pieces_size array
        new_obj.pieces_size[:] = self.pieces_size
        return new_obj

    @cython.ccall
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.locals(piece_index=cython.int, color=cython.int)
    def play_piece(self, piece_index, color):
        assert 0 <= self.rem_pieces[color - 1, piece_index] <= 1
        if self.rem_pieces[color - 1, piece_index] >= 1:
            self.rem_pieces[color - 1, piece_index] = False
            self.pieces_count[color - 1] -= 1
            self.pieces_size[color - 1] -= piece_sizes[piece_index]
        else:
            raise ValueError(f"Player {color} doesn't have piece {piece_index} available.")

    @cython.ccall
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.locals(piece_index=cython.int, color=cython.int, color_i=cython.int, piece_i=cython.int)
    def avail_pieces_for_color(self, color) -> cython.list:
        return [
            piece_index
            for piece_index in range(self.rem_pieces.shape[1])
            if self.rem_pieces[color - 1, piece_index] >= 1
        ]

    @cython.ccall
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.locals(color=cython.int)
    def pieces_left_for_color(self, color) -> cython.int:
        return self.pieces_count[color - 1]

    @cython.ccall
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.locals(player=cython.int)
    def pieces_left_for_player(self, player) -> cython.int:
        if player == 1:
            return self.pieces_count[0] + self.pieces_count[2]
        else:
            return self.pieces_count[1] + self.pieces_count[3]

    @cython.ccall
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.locals(player=cython.int)
    def sum_piece_size(self, player) -> cython.int:
        if player == 1:
            return self.pieces_size[0] + self.pieces_size[2]
        else:
            return self.pieces_size[1] + self.pieces_size[3]

    def __str__(self):
        return f"BlokusPieces:\n Pieces count: {self.pieces_count}"

    def visualize(self) -> str:
        output = []
        for player in [1, 2]:
            output.append(f"Player {player}:")
            for color in [player, player + 2]:  # assuming color 1 and 3 for player 1, 2 and 4 for player 2
                pieces_count = self.pieces_left_for_color(color)
                total_size = self.sum_piece_size(player)
                available_pieces = ", ".join(str(i) for i in self.avail_pieces_for_color(color))
                output.append(
                    f"\tColor {colors[color - 1]}: Pieces left: {pieces_count}, Total size: {total_size}"
                )
                output.append(f"\tAvailable pieces: {available_pieces}")
        return "\n".join(output)


@cython.cclass
class BlokusGameState(GameState):
    # Changed zobrist_table size to include 4 players
    zobrist_table = np.random.randint(
        low=0,
        high=np.iinfo(np.int64).max,
        size=(BOARD_SIZE, BOARD_SIZE, 5),  # 4 players + 1 for empty state
        dtype=np.int64,
    )
    REUSE = True

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
        return calculate_board_hash(self.board, self.zobrist_table)

    def skip_turn(self):
        """Used for the null-move heuristic in alpha-beta search"""
        new_state = BlokusGameState(
            board=np.copy(self.board),
            pieces=self.pieces.custom_copy(),
            player=3 - self.player,
            n_turns=self.n_turns + 1,
            passed=self.passed.copy(),
        )
        new_state.board_hash = new_state._calculate_board_hash()
        return new_state

    def apply_action(self, action: Tuple[int, int, int, int]) -> BlokusGameState:
        new_state = BlokusGameState(
            board=np.copy(self.board),
            pieces=self.pieces.custom_copy(),
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

        return get_random_legal_action(
            self.board, self.color, self.n_turns, self.pieces.avail_pieces_for_color(self.color)
        )

    def yield_legal_actions(self) -> Generator[Tuple[int, int, int, int], None, None]:
        if self.passed[self.color]:
            yield PASS_MOVE
            return

        legal_actions = get_legal_actions(
            self.board, self.color, self.n_turns, self.pieces.avail_pieces_for_color(self.color)
        )

        for action in legal_actions:
            yield action

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
        scores: list[tuple] = [()] * len(moves)
        for i in range(len(moves)):
            scores[i] = (
                moves[i],
                evaluate_move(
                    self.board, moves[i][0], moves[i][1], moves[i][2], moves[i][3], self.player, self.n_turns
                ),
            )

        return scores

    def move_weights(self, moves):
        """
        Evaluates a list of moves, preferring the placement of larger pieces first.

        :param moves: The list of moves to evaluate.
        :return: The list of scores for the moves.
        """
        scores: list[int] = [0] * len(moves)
        for i in range(len(moves)):
            scores[i] = evaluate_move(
                self.board, moves[i][0], moves[i][1], moves[i][2], moves[i][3], self.player, self.n_turns
            )
        return scores

    def evaluate_move(self, move):
        # Prefer placing larger pieces first
        # _, _, piece_index, _ = move
        # if piece_index != -1:  # In the case of a pass move
        #     return piece_sizes[piece_index]  # This works because piece cells are represented by 1's
        # return 0
        return evaluate_move(self.board, move[0], move[1], move[2], move[3], self.player, self.n_turns)

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

    def visualize(self, full_debug=False) -> str:
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

        output = "\n".join(" ".join(cell for cell in row) for row in visual_board) + "\n"

        # Include additional debug info if full_debug is True
        if full_debug:
            actions = self.get_legal_actions()

            debug_info = f"""\n
            Player: {self.player}, Color: {colors[self.color-1]} (1: [{colors[0]}, {colors[2]}], 2: [{colors[1]},{colors[2]}])
            Reward: {self.get_reward(1)}/{self.get_reward(2)}
            Terminal: {self.is_terminal()} 
            n_turns: {self.n_turns} \n
            {len(actions)} actions available
            passed: {self.passed} \n
            {"***" * 10} \n
            eval ({evaluate_blokus(self, 1)}/{evaluate_blokus(self, 2)}) \n
            {"***" * 10} \n
            {self.pieces.visualize()}
            """
            output += debug_info

            if len(actions) > 0:
                actions = self.evaluate_moves(self.get_legal_actions())
                actions = sorted(actions, key=lambda x: x[1], reverse=True)
                output += "\n" + "..." * 60
                output += "\n" + str(actions[:20])

        return output

    def __repr__(self) -> str:
        return "blokus"

    @property
    def transposition_table_size(self):
        return 2 ** (BOARD_SIZE)


@cython.ccall
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.locals(
    x=cython.int,
    y=cython.int,
    piece_index=cython.int,
    rotation=cython.int,
    player=cython.int,
    o_color1=cython.int,
    o_color2=cython.int,
    indices=cython.int[:, :],
    score=cython.int,
    new_i=cython.int,
    new_j=cython.int,
    di=cython.int,
    dj=cython.int,
    ortho_i=cython.int,
    ortho_j=cython.int,
    turns=cython.int,
)
def evaluate_move(board: cython.int[:, :], x, y, piece_index, rotation, player, turns) -> cython.int:
    if piece_index == -1:  # In the case of a pass move
        return 0

    score = int(piece_sizes[piece_index] ** 2)

    if turns > 12:  # don't start checking until there are sufficient pieces on the board
        if player == 1:
            o_color1 = 2
            o_color2 = 4
        else:
            o_color1 = 1
            o_color2 = 3

        indices: cython.int[:, :] = piece_indices[piece_index][rotation]

        for ind in range(indices.shape[0]):
            new_i = indices[ind, 0] + x
            new_j = indices[ind, 1] + y

            # Check orthogonals if we are touching an opponent, that's good!
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if abs(di) != abs(dj):
                        ortho_i, ortho_j = new_i + di, new_j + dj
                        # If the orthogonal cell is inside the board and touches the opponent's color
                        if (
                            0 <= ortho_i < 20
                            and 0 <= ortho_j < 20
                            and board[ortho_i, ortho_j] == o_color1
                            or board[ortho_i, ortho_j] == o_color2
                        ):
                            score += 10

    return score


@cython.ccall
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.locals(x=cython.int, y=cython.int, player=cython.int)
def calculate_board_hash(board: cython.int[:, :], zobrist_table: cython.long[:, :, :]) -> cython.long:
    board_hash: cython.uint = 0
    for x in range(20):
        for y in range(20):
            player = int(board[x, y])
            board_hash ^= zobrist_table[x, y, player]
    return board_hash


@cython.ccall
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def update_state(
    board: cython.int[:, :],
    x: cython.int,
    y: cython.int,
    piece_index: cython.int,
    rotation: cython.int,
    color: cython.int,
):
    indices: cython.int[:, :] = piece_indices[piece_index][rotation]
    i: cython.int
    dx: cython.int
    dy: cython.int

    for i in range(indices.shape[0]):
        dx = indices[i, 0]
        dy = indices[i, 1]
        board[x + dx, y + dy] = color


@cython.ccall
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
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

                        if is_legal_action(x, y, piece_index, rotation, n_turns, color, board):
                            legal_actions.append(action)

    if not legal_actions:  # If legal_actions list is empty, then it's a PASS_MOVE
        return [PASS_MOVE]

    return legal_actions


@cython.ccall
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
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
)
def get_random_legal_action(
    board: cython.int[:, :],
    color: cython.int,
    n_turns: cython.int,
    avail_pieces: cython.list,
    attempts: cython.int = 1000,
) -> tuple[cython.int, cython.int, cython.int, cython.int]:
    checked_actions: cython.set = set()
    # Get a set of points around which to look, preventing us from checking the entire board for each piece
    if n_turns > 3:
        perimeter_points: set = find_corners_for_color(board, color)
    else:
        perimeter_points = BOARD_CORNERS

    while len(perimeter_points) > 0:
        px, py = perimeter_points.pop()
        for _ in range(attempts):
            piece_index = avail_pieces[c_random(0, len(avail_pieces) - 1)]
            rotation = c_random(0, unique_rotations[piece_index] - 1)  # Use pre-computed rotations
            piece = rotated_pieces[piece_index][rotation]
            piece_width = piece.shape[0]
            piece_height = piece.shape[1]

            dx = c_random(max(-piece_width + 1, -px), min(piece_width, 20 - px))
            dy = c_random(max(-piece_height + 1, -py), min(piece_height, 20 - py))

            x, y = px + dx, py + dy
            # Check if piece can be physically placed, considering its shape
            if not (0 <= x + piece_width - 1 < 20 and 0 <= y + piece_height - 1 < 20):
                continue

            action = (x, y, piece_index, rotation)

            if action in checked_actions:
                continue

            checked_actions.add(action)

            if is_legal_action(x, y, piece_index, rotation, n_turns, color, board):
                return action
    assert False, "No legal action found"
    return PASS_MOVE  # If no legal move found after the given number of attempts, return a pass move


@cython.ccall
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
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
    piece_index: cython.int,
    rotation: cython.int,
    n_turns: cython.int,
    color: cython.int,
    board: cython.int[:, :],
) -> cython.bint:
    indices: cython.int[:, :] = piece_indices[piece_index][rotation]

    # Verify if it's the first piece and if it's on the corner
    if n_turns <= 3 and is_on_board_corner(x, y, rotated_pieces[piece_index][rotation]):
        return no_overlap(board, x, y, indices)
    elif n_turns <= 3:
        return False

    # Piece should not overlap with an existing piece or opponent's piece
    if not no_overlap(board, x, y, indices):
        return False

    corner_touch: cython.bint = 0
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
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.locals(board=cython.int[:, :], x=cython.int, y=cython.int, i=cython.int)
def no_overlap(board, x, y, indices: cython.int[:, :]) -> cython.bint:
    for i in range(indices.shape[0]):
        if board[x + indices[i, 0], y + indices[i, 1]] != 0:
            return 0  # Found overlap, so return False
    return 1  # No overlap was found, so return True


@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
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


@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.locals(
    board=cython.int[:, :],
    color=cython.int,
    i=cython.int,
    j=cython.int,
    dx=cython.int,
    dy=cython.int,
    di=cython.int,
    dj=cython.int,
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
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.locals(
    game_state=cython.object,
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
    player,
    m_corn_diff=10,
    m_piece_size=2,
    m_turn=0.9,
    a=50,
    norm=0,
) -> cython.double:
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

    player_corners = len(find_corners_for_color(board, p_color1)) + len(
        find_corners_for_color(board, p_color2)
    )
    opponent_corners = len(find_corners_for_color(board, o_color1)) + len(
        find_corners_for_color(board, o_color2)
    )

    # More corners means more opportunities to move
    corner_diff = player_corners - opponent_corners
    # Prefer to place pieces that are as large as possible
    player_pieces_sizes = state_pieces.sum_piece_size(player)
    opponent_pieces_sizes = state_pieces.sum_piece_size(opponent)

    piece_size_diff = opponent_pieces_sizes - player_pieces_sizes

    score = m_corn_diff * corner_diff + m_piece_size * piece_size_diff

    score = score if game_state.player == player else m_turn * score

    if norm:
        return normalize(score, a)
    else:
        return score

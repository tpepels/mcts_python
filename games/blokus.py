# cython: language_level=3
# distutils: language=c++

import array

import cython
from cython.cimports.libcpp.set import set as cset
from cython.cimports.libcpp.pair import pair
from cython.cimports.libcpp.vector import vector
from cython.cimports import numpy as cnp
from cython.cimports.includes import c_random, normalize, GameState, draw, loss, win
from cython.cimports.games.blokus import (
    BOARD_SIZE,
    PASS_MOVE,
    PIECE_SIZES,
    UNIQUE_ROTATIONS,
    BOARD_CORNERS,
)

cnp.import_array()
import numpy as np
from termcolor import colored

if cython.compiled:
    print("Blokus is compiled.")
else:
    print("Blokus is just a lowly interpreted script.")

BOARD_SIZE = 20
PASS_MOVE = (-1, -1, -1, -1)
BOARD_CORNERS.insert(pair[cython.int, cython.int](0, 0))
BOARD_CORNERS.insert(pair[cython.int, cython.int](0, BOARD_SIZE - 1))
BOARD_CORNERS.insert(
    pair[cython.int, cython.int](BOARD_SIZE - 1, 0),
)
BOARD_CORNERS.insert(pair[cython.int, cython.int](BOARD_SIZE - 1, BOARD_SIZE - 1))


PIECES: cython.list = [
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


@cython.ccall
@cython.profile(False)
def precompute():
    global PIECE_SIZES, UNIQUE_ROTATIONS, ROTATED_PIECES, PIECE_INDICES
    # Pre-compute the number of 1's in each piece
    piece_index: cython.int
    piece: cnp.ndarray
    for piece_index, piece in enumerate(PIECES):
        PIECE_SIZES[piece_index] = np.sum(piece)

    rotations: cython.list
    flips: cython.list
    unique: cython.list
    rotated: cnp.ndarray
    u: cnp.ndarray

    for piece_index, piece in enumerate(PIECES):
        rotations = [np.rot90(piece, rotation) for rotation in range(4)]
        # Add flipped versions of the piece and its rotations.
        flips = [np.flip(piece), np.flip(piece, 0), np.flip(piece, 1)]
        rotations += flips

        unique = [piece.tolist()]  # initialize with original piece

        for rotated in rotations:
            found = False
            for un in unique:
                if np.array_equal(rotated, np.array(un)):
                    found = True
                    break

            if not found:
                unique.append(rotated)

        UNIQUE_ROTATIONS[piece_index] = len(unique)

    ROTATED_PIECES = [None] * len(PIECES)
    PIECE_INDICES = [None] * len(PIECES)
    for piece_index, piece in enumerate(PIECES):
        rotations = [np.rot90(piece, rotation) for rotation in range(UNIQUE_ROTATIONS[piece_index])]
        # Add flipped versions of the piece and its rotations.
        flips = [np.flip(piece), np.flip(piece, 0), np.flip(piece, 1)]
        rotations += flips

        ROTATED_PIECES[piece_index] = [np.array(rotation).astype(np.int32) for rotation in rotations]
        PIECE_INDICES[piece_index] = [np.argwhere(rotation).astype(np.int32) for rotation in rotations]


precompute()


@cython.cclass
class BlokusPieces:
    rem_pieces = cython.declare(cython.short[:, :])
    pieces_count = cython.declare(cython.short[:])
    pieces_size = cython.declare(cython.short[:])

    def __init__(self, init_state=False):
        if init_state:
            n_pieces = len(PIECES)
            # Initialized all pieces to be available (True)
            self.rem_pieces = np.ones((4, n_pieces), dtype=np.int16)
            # Initialized all counts to be the total number of pieces
            self.pieces_count = np.full(4, n_pieces, dtype=np.int16)
            # Initialize all sizes to the total size of pieces
            total_size = np.sum([np.sum(piece) for piece in PIECES])
            self.pieces_size = np.full(4, total_size, dtype=np.int16)

    @cython.ccall
    @cython.locals(new_obj=BlokusPieces)
    def custom_copy(self) -> BlokusPieces:
        new_obj = BlokusPieces(init_state=False)
        # Copying rem_pieces array
        new_obj.rem_pieces = self.rem_pieces.copy()
        # Copying pieces_count array
        new_obj.pieces_count = self.pieces_count.copy()
        # Copying pieces_size array
        new_obj.pieces_size = self.pieces_size.copy()
        return new_obj

    @cython.ccall
    @cython.locals(piece_index=cython.int, color=cython.int)
    def play_piece(self, piece_index, color):
        assert 0 <= self.rem_pieces[color - 1, piece_index] <= 1
        if self.rem_pieces[color - 1, piece_index] >= 1:
            self.rem_pieces[color - 1, piece_index] = False
            self.pieces_count[color - 1] -= 1
            self.pieces_size[color - 1] -= PIECE_SIZES[piece_index]
        else:
            raise ValueError(f"Player {color} doesn't have piece {piece_index} available.")

    @cython.ccall
    @cython.locals(piece_index=cython.int, color=cython.int, color_i=cython.int, piece_i=cython.int)
    def avail_pieces_for_color(self, color) -> cython.list:
        return [
            piece_index
            for piece_index in range(self.rem_pieces.shape[1])
            if self.rem_pieces[color - 1, piece_index] >= 1
        ]

    @cython.ccall
    @cython.exceptval(-1, check=False)
    @cython.locals(color=cython.int)
    def pieces_left_for_color(self, color) -> cython.int:
        return self.pieces_count[color - 1]

    @cython.ccall
    @cython.exceptval(-1, check=False)
    @cython.locals(player=cython.int)
    def pieces_left_for_player(self, player) -> cython.int:
        if player == 1:
            return self.pieces_count[0] + self.pieces_count[2]
        else:
            return self.pieces_count[1] + self.pieces_count[3]

    @cython.ccall
    @cython.exceptval(-1, check=False)
    @cython.locals(player=cython.int)
    def sum_piece_size(self, player) -> cython.int:
        if player == 1:
            return self.pieces_size[0] + self.pieces_size[2]
        else:
            return self.pieces_size[1] + self.pieces_size[3]

    def __str__(self):
        return f"BlokusPieces:\n Pieces count: {self.pieces_count}"

    def visualize(self) -> str:
        colors = ["green", "red", "blue", "yellow"]
        output = []
        for player in [1, 2]:
            pstr = ""
            for color in [player, player + 2]:  # assuming color 1 and 3 for player 1, 2 and 4 for player 2
                pieces_count = self.pieces_left_for_color(color)
                total_size = self.sum_piece_size(player)
                pstr += f"| {player} ({colors[color - 1]}) pieces left: {pieces_count}, size: {total_size} |"

            output.append(pstr)
        return "\n".join(output)


@cython.cclass
class BlokusGameState(GameState):
    """
    Blokus game state.
    """

    # Changed zobrist_table size to include 4 players
    table_values = np.random.randint(
        low=0,
        high=np.iinfo(np.int64).max,
        size=(BOARD_SIZE, BOARD_SIZE, 5),  # 4 players + 1 for empty state
        dtype=np.int64,
    )
    REUSE = True

    # Public variables
    # player = cython.declare(cython.int, visibility="public")
    color = cython.declare(cython.int, visibility="public")
    board = cython.declare(cython.int[:, :], visibility="public")
    board_hash = cython.declare(cython.longlong, visibility="public")
    last_action = cython.declare(cython.tuple, visibility="public")
    # Private variables
    passed: cython.unsignedshort[:]
    pieces: BlokusPieces
    positions_checked: cython.int
    n_turns: cython.int
    zobrist_table: cython.longlong[:, :, :]

    def __init__(
        self, board=None, pieces=None, player=1, n_turns=0, passed=None, last_action=(-1, -1, -1, -1)
    ):
        if passed is None:
            passed = np.zeros((4,), dtype=np.uint8)

        self.passed = passed

        self.zobrist_table = self.table_values

        if pieces is None:
            self.pieces = BlokusPieces(init_state=True)
        else:
            self.pieces = pieces

        self.player = player

        if board is not None:
            self.board = board
        else:
            self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
            self._calculate_board_hash()  # Works in-placen ot return

        self.n_turns = n_turns
        self.color = (n_turns % 4) + 1

        self.positions_checked = 0
        self.last_action = last_action

    @cython.cfunc
    @cython.inline
    @cython.locals(x=cython.int, y=cython.int)
    def _calculate_board_hash(self) -> cython.void:
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                self.board_hash ^= self.zobrist_table[x, y, self.board[x, y]]

    @cython.cfunc
    def skip_turn(self) -> BlokusGameState:
        """Used for the null-move heuristic in alpha-beta search"""
        new_state: BlokusGameState = BlokusGameState(
            board=self.board.copy(),
            pieces=self.pieces.custom_copy(),
            player=3 - self.player,
            n_turns=self.n_turns + 1,
            passed=self.passed.copy(),
        )
        new_state._calculate_board_hash()
        return new_state

    @cython.cfunc
    @cython.locals(
        x=cython.int,
        y=cython.int,
        piece_index=cython.int,
        rotation=cython.int,
        i=cython.int,
        x=cython.int,
        dx=cython.int,
        dy=cython.int,
    )
    def apply_action_playout(self, action: cython.tuple) -> cython.void:
        t_action: cython.tuple[cython.int, cython.int, cython.int, cython.int] = action
        if t_action[0] == -1:
            self.passed[self.color - 1] = 1
        else:
            assert self.passed[self.color - 1] == 0, "Color has already passed"

            x = t_action[0]
            y = t_action[1]
            piece_index = t_action[2]
            rotation = t_action[3]

            indices: cython.int[:, :] = PIECE_INDICES[piece_index][rotation]

            for i in range(indices.shape[0]):
                dx = indices[i, 0]
                dy = indices[i, 1]
                self.board[x + dx, y + dy] = self.color

            self.pieces.play_piece(piece_index, self.color)

        self.n_turns += 1
        self.color = (self.n_turns % 4) + 1
        self.player = 3 - self.player
        self.last_action = action

    @cython.ccall
    @cython.locals(
        x=cython.int,
        y=cython.int,
        piece_index=cython.int,
        rotation=cython.int,
        i=cython.int,
        dx=cython.int,
        dy=cython.int,
    )
    def apply_action(self, action: cython.tuple) -> BlokusGameState:
        new_state: BlokusGameState = BlokusGameState(
            board=self.board.copy(),
            pieces=self.pieces.custom_copy(),
            player=self.player,
            n_turns=self.n_turns,
            passed=self.passed.copy(),
            last_action=action,
        )
        t_action: cython.tuple[cython.int, cython.int, cython.int, cython.int] = action

        if t_action[0] == -1:
            new_state.passed[self.color - 1] = True
        else:
            assert not new_state.passed[self.color - 1], "Color has already passed"
            x = t_action[0]
            y = t_action[1]
            piece_index = t_action[2]
            rotation = t_action[3]

            indices: cython.int[:, :] = PIECE_INDICES[piece_index][rotation]

            for i in range(indices.shape[0]):
                dx = indices[i, 0]
                dy = indices[i, 1]
                new_state.board[x + dx, y + dy] = self.color

            new_state.pieces.play_piece(piece_index, self.color)

        new_state.n_turns += 1
        new_state.color = (new_state.n_turns % 4) + 1
        new_state.player = 3 - self.player
        # For this game it's quicker and much simpler to recalcuate the hash than to update it
        new_state._calculate_board_hash()
        return new_state

    @cython.cfunc
    @cython.locals(
        px=cython.int,
        py=cython.int,
        dx=cython.int,
        dy=cython.int,
        x=cython.int,
        y=cython.int,
        action=cython.tuple[cython.int, cython.int, cython.int, cython.int],
        ppair=pair[cython.int, cython.int],
        perimeter_points=cset[pair[cython.int, cython.int]],
    )
    def get_random_action(self) -> cython.tuple:
        avail_pieces: cython.list = self.pieces.avail_pieces_for_color(self.color)

        if self.passed[self.color - 1] == 1 or len(avail_pieces) == 0:
            return PASS_MOVE

        checked_actions: cython.set = set()
        # Get a set of points around which to look, preventing us from checking the entire board for each piece
        if self.n_turns > 3:
            perimeter_points = find_corners_for_color_cpp(self.board, self.color)
        else:
            perimeter_points = BOARD_CORNERS

        for ppair in perimeter_points:
            px, py = ppair.first, ppair.second
            for _ in range(100):  # ! TODO Check this number
                piece_index: cython.int = avail_pieces[c_random(0, len(avail_pieces) - 1)]
                rotation: cython.int = c_random(0, UNIQUE_ROTATIONS[piece_index] - 1)
                piece: cython.int[:, :] = ROTATED_PIECES[piece_index][rotation]
                piece_width: cython.int = piece.shape[0]
                piece_height: cython.int = piece.shape[1]

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

                if is_legal_action(x, y, piece_index, rotation, self.n_turns, self.color, self.board):
                    return action

        return PASS_MOVE  # If no legal move found after the given number of attempts, return a pass move

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
        perimeter_points=cset[pair[cython.int, cython.int]],
    )
    def get_legal_actions(self) -> cython.list:
        avail_pieces: cython.list = self.pieces.avail_pieces_for_color(self.color)

        if self.passed[self.color - 1] == 1 or len(avail_pieces) == 0:
            return [PASS_MOVE]

        legal_actions: cython.list = []
        checked_actions: cython.set = set()

        # Get a set of points around which to look, preventing us from checking the entire board for each piece
        if self.n_turns > 3:
            perimeter_points = find_corners_for_color_cpp(self.board, self.color)
        else:
            perimeter_points = BOARD_CORNERS

        for ppair in perimeter_points:
            px, py = ppair.first, ppair.second
            for p in range(len(avail_pieces)):
                piece_index = avail_pieces[p]
                for rotation in range(UNIQUE_ROTATIONS[piece_index]):  # Use pre-computed rotations
                    piece = ROTATED_PIECES[piece_index][rotation]
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

                            if is_legal_action(
                                x, y, piece_index, rotation, self.n_turns, self.color, self.board
                            ):
                                legal_actions.append(action)

        if not legal_actions:  # If legal_actions list is empty, then it's a PASS_MOVE
            return [PASS_MOVE]

        return legal_actions

    param_order: dict = {"m_corn_diff": 0, "m_piece_size": 1, "m_turn": 2, "a": 3}
    default_params = array.array("d", [10, 2, 0.9, 150])

    @cython.cfunc
    @cython.exceptval(-9999999, check=False)
    def evaluate(
        self, player: cython.int, params: cython.double[:], norm: cython.bint = False
    ) -> cython.double:
        if player == 1:
            p_color1: cython.int = 1
            p_color2: cython.int = 3
            o_color1: cython.int = 2
            o_color2: cython.int = 4
        else:
            p_color1: cython.int = 2
            p_color2: cython.int = 4
            o_color1: cython.int = 1
            o_color2: cython.int = 3

        # Try to have as many corners as possible
        corner_diff: cython.int = count_corners(self.board, p_color1) + count_corners(self.board, p_color2)
        corner_diff -= count_corners(self.board, o_color1) + count_corners(self.board, o_color2)

        # Prefer to place pieces that are as large as possible
        piece_size_diff: cython.int = self.pieces.sum_piece_size(3 - player) - self.pieces.sum_piece_size(
            player
        )
        # "m_corn_diff": 0, "m_piece_size": 1, "m_turn": 2, "a": 3
        score: cython.double = (params[0] * corner_diff) + (params[1] * piece_size_diff)
        # If it's the player's turn, then multiply by m_turn
        score *= 1 if self.player == player else params[2]

        if norm:
            return normalize(score, params[3])
        else:
            return score

    @cython.cfunc
    @cython.locals(i=cython.int)
    def evaluate_moves(self, moves: cython.list) -> cython.list:
        """
        Evaluates a list of moves, preferring the placement of larger pieces first.

        :param moves: The list of moves to evaluate.
        :return: The list of scores for the moves.
        """
        scores: cython.list = [()] * len(moves)
        for i in range(len(moves)):
            scores[i] = (moves[i], self.evaluate_move(moves[i]))

        return scores

    @cython.cfunc
    @cython.locals(i=cython.int)
    def move_weights(self, moves: cython.list) -> cython.list:
        """
        Evaluates a list of moves, preferring the placement of larger pieces first.

        :param moves: The list of moves to evaluate.
        :return: The list of scores for the moves.
        """
        n_moves: cython.int = len(moves)
        scores: vector[cython.int]
        scores.reserve(n_moves)
        for i in range(n_moves):
            move: cython.tuple = moves[i]
            scores.push_back(self.evaluate_move(move))

        return scores

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    @cython.locals(
        x=cython.int,
        y=cython.int,
        piece_index=cython.int,
        rotation=cython.int,
        ind=cython.int,
        di=cython.int,
        dj=cython.int,
        ortho_i=cython.int,
        ortho_j=cython.int,
        o_color1=cython.int,
        o_color2=cython.int,
        score=cython.int,
        indices=cython.int[:, :],
    )
    def evaluate_move(self, move: cython.tuple) -> cython.int:
        t_move: cython.tuple[cython.int, cython.int, cython.int, cython.int] = move

        x, y, piece_index, rotation = t_move

        if piece_index == -1:  # In the case of a pass move
            return 0

        score = int(PIECE_SIZES[piece_index] ** 2)

        if self.n_turns > 12:  # don't start checking until there are sufficient pieces on the board
            if self.player == 1:
                o_color1: cython.int = 2
                o_color2: cython.int = 4
            else:
                o_color1: cython.int = 1
                o_color2: cython.int = 3

            indices: cython.int[:, :] = PIECE_INDICES[piece_index][rotation]

            for ind in range(indices.shape[0]):
                new_i: cython.int = indices[ind, 0] + x
                new_j: cython.int = indices[ind, 1] + y

                # Check orthogonals if we are touching an opponent, that's good!
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if abs(di) != abs(dj):
                            ortho_i, ortho_j = new_i + di, new_j + dj
                            # If the orthogonal cell is inside the board and touches the opponent's color
                            if (0 <= ortho_i < 20 and 0 <= ortho_j < 20) and (
                                self.board[ortho_i, ortho_j] == o_color1
                                or self.board[ortho_i, ortho_j] == o_color2
                            ):
                                score += 10
        return score

    @cython.ccall
    def is_terminal(self) -> cython.bint:
        # If both players have passed, then we stop the game.
        if self.passed[0] == 1 and self.passed[1] == 1 and self.passed[2] == 1 and self.passed[3] == 1:
            return 1

        return 0

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def get_reward(self, player: cython.int) -> cython.int:
        if not self.is_terminal():
            return 0

        player_pieces_left: cython.int = self.pieces.sum_piece_size(player)
        opponent_pieces_left: cython.int = self.pieces.sum_piece_size(3 - player)

        if opponent_pieces_left > player_pieces_left:
            return win
        elif opponent_pieces_left < player_pieces_left:
            return loss
        else:
            return draw

    @cython.cfunc
    def get_result_tuple(self) -> cython.tuple:
        result: cython.int = self.get_reward(1)
        if result == win:
            return (1.0, 0.0)
        elif result == loss:
            return (0.0, 1.0)

        return (0.5, 0.5)

    @cython.cfunc
    def is_capture(self, move: cython.tuple) -> cython.bint:
        # Not applicable for this game as there are no captures
        return False

    @cython.ccall
    def visualize(self, full_debug=False) -> str:
        colors = ["green", "red", "blue", "yellow"]
        visual_board = []

        perimeters = {}
        for i in range(4):
            perimeters[i] = find_corners_for_color_cpp(self.board, i + 1)

        # Add a border to the top
        border_row = []
        for _ in range(len(self.board[0]) + 2):
            border_row.append("#")

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

        output = ""
        for row in visual_board:
            row_string = ""
            for cell in row:
                row_string += cell + " "
            row_string = row_string.rstrip()  # Remove trailing space
            output += row_string + "\n"

        # Include additional debug info if full_debug is True
        if full_debug:
            actions = self.get_legal_actions()

            debug_info = "\n"
            debug_info += f"Player: {self.player}, Color: {colors[self.color-1]} (1: [{colors[0]}, {colors[2]}], 2: [{colors[1]},{colors[2]}]) \n"
            debug_info += f"Reward: {self.get_reward(1)}/{self.get_reward(2)} | Terminal: {self.is_terminal()} | n_turns: {self.n_turns} | {len(actions)} actions available \n"
            debug_info += (
                f"passed: 1:{self.passed[0]}, 2:{self.passed[1]}, 3:{self.passed[2]}, 4:{self.passed[3]} \n"
            )
            debug_info += "---" * 20 + "\n"
            debug_info += f"eval ({self.evaluate(1, self.default_params, norm=False):.2f}/{self.evaluate(2, self.default_params, norm=False):.2f}) | "
            debug_info += f"norm ({self.evaluate(1, self.default_params, norm=True):.2f}/{self.evaluate(2, self.default_params, norm=True):.2f}) \n"
            debug_info += "---" * 20 + "\n"
            debug_info += f"{self.pieces.visualize()}"

            output += debug_info

            # if len(actions) > 0:
            #     actions = self.evaluate_moves(self.get_legal_actions())
            #     actions = sorted(actions, key=lambda x: x[1], reverse=True)
            #     output += "\n" + "..." * 60
            #     output += "\n" + str(actions[:20])

        return output

    def __repr__(self) -> str:
        return "blokus"

    @property
    def transposition_table_size(self) -> cython.int:
        return cython.cast(cython.int, 2 ** (BOARD_SIZE))


@cython.cfunc
@cython.inline
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
    indices: cython.int[:, :] = PIECE_INDICES[piece_index][rotation]

    # Verify if it's the first piece and if it's on the corner
    if n_turns <= 3 and is_on_board_corner(x, y, indices):
        return no_overlap(board, x, y, indices)
    elif n_turns <= 3:
        return 0

    # Piece should not overlap with an existing piece or opponent's piece
    if not no_overlap(board, x, y, indices):
        return 0

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
                    return 0  # Some cell of the piece is orthogonally touching a cell of an existing piece of the player

    return corner_touch


@cython.cfunc
@cython.inline
@cython.exceptval(-1, check=False)
@cython.locals(board=cython.int[:, :], x=cython.int, y=cython.int, i=cython.int, indices=cython.int[:, :])
def no_overlap(board, x, y, indices) -> cython.bint:
    for i in range(indices.shape[0]):
        if board[x + indices[i, 0], y + indices[i, 1]] != 0:
            return 0  # Found overlap, so return False
    return 1  # No overlap was found, so return True


# @cython.cfunc
# @cython.inline
# @cython.locals(
#     piece_x=cython.int,
#     piece_y=cython.int,
#     i=cython.int,
#     j=cython.int,
#     piece_width=cython.int,
#     piece_height=cython.int,
# )
# def is_on_board_corner(x: cython.int, y: cython.int, piece: cython.int[:, :]) -> cython.bint:
#     piece_width = piece.shape[0]
#     piece_height = piece.shape[1]
#     # Check if any part of the piece coincides with any of the board's corners
#     for i in range(piece_width):
#         for j in range(piece_height):
#             if piece[i, j]:  # If this part of the piece array contains the piece
#                 piece_x = x + i
#                 piece_y = y + j
#                 # Check if the piece is at any of the board's corners
#                 if (piece_x == 0 or piece_x == 19) and (piece_y == 0 or piece_y == 19):
#                     return 1
#     return 0


@cython.cfunc
@cython.inline
@cython.exceptval(-1, check=False)
@cython.locals(
    x=cython.int, y=cython.int, indices=cython.int[:, :], i=cython.int, piece_x=cython.int, piece_y=cython.int
)
def is_on_board_corner(x, y, indices) -> cython.bint:
    for i in range(indices.shape[0]):
        piece_x = x + indices[i, 0]
        piece_y = y + indices[i, 1]
        # Check if the piece is at any of the board's corners
        if (piece_x == 0 or piece_x == 19) and (piece_y == 0 or piece_y == 19):
            return 1
    return 0


""" 
@cython.cfunc
@cython.inline
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
    coßrner_points: cython.set = set()

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
 """


@cython.cfunc
@cython.inline
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
    width=cython.int,
    height=cython.int,
    start_i=cython.int,
    start_j=cython.int,
    j_offset=cython.int,
    i_offset=cython.int,
)
def find_corners_for_color_cpp(board, color) -> cset[pair[cython.int, cython.int]]:
    corner_points: cset[pair[cython.int, cython.int]]

    # Get board dimensions
    height, width = board.shape[0], board.shape[1]

    # Generate random start points
    start_i = c_random(0, height - 1)
    start_j = c_random(0, width - 1)

    # Loop from the random start point, wrapping around
    for i_offset in range(height):
        for j_offset in range(width):
            i = (start_i + i_offset) % height
            j = (start_j + j_offset) % width
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
                                corner_points.insert(pair[cython.int, cython.int](new_i, new_j))

    return corner_points


@cython.cfunc
@cython.inline
@cython.exceptval(-1, check=False)
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
def count_corners(board, color) -> cython.int:
    corner_points: cython.int = 0

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
                                corner_points += 1
    return corner_points

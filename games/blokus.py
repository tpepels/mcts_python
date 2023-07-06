import copy
from typing import List, Tuple
import numpy as np

from games.gamestate import GameState, win, loss, draw

PLAYER_COUNT = 2
BOARD_SIZE = 20

PASS_MOVE = (-1, -1, -1, -1)
START_POSITIONS = [(0, 0), (BOARD_SIZE - 1, BOARD_SIZE - 1)]  # corner positions
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
DIAGONALS = [(1, 1), (1, -1), (-1, -1), (-1, 1)]


class BlokusGameState(GameState):
    def __init__(self, board=None, pieces=None, player=1, n_turns=0, passes=0):
        if board is None:
            self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        else:
            self.board = board

        if pieces is None:
            self.pieces = BlokusPieces()
        else:
            self.pieces = pieces
        self.player = player
        self.n_turns = n_turns
        self.passes = passes

    def copy(self) -> "BlokusGameState":
        return BlokusGameState(
            board=np.copy(self.board),
            pieces=self.pieces.copy(),
            player=self.player,
            n_turns=self.n_turns,
            passes=self.passes,
        )

    def apply_action(self, action: Tuple[int, int, int, int]) -> "BlokusGameState":
        new_state = self.copy()
        # Pass move if player cannot make a move anymore
        if action == PASS_MOVE:
            new_state.passes += 1
        else:
            x, y, piece_index, rotation = action
            piece = np.rot90(self.pieces.pieces[piece_index], rotation)
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
            new_state.n_turns += 1
            new_state.pieces.play_piece(piece_index, self.player + 1)

        new_state.player = 3 - self.player
        return new_state

    def get_legal_actions(self) -> List[Tuple[int, int, int, int]]:
        legal_actions = []
        for piece_index in self.pieces.get_available_pieces(self.player + 1):
            for rotation in range(4):
                rotated_piece = np.rot90(self.pieces.pieces[piece_index], rotation)
                piece_width, piece_height = rotated_piece.shape
                for x in range(BOARD_SIZE - piece_width + 1):
                    for y in range(BOARD_SIZE - piece_height + 1):
                        if self.is_legal_action((x, y, piece_index, rotation)):
                            legal_actions.append((x, y, piece_index, rotation))

        if legal_actions == []:
            return [
                PASS_MOVE,
            ]
        return legal_actions

    def is_legal_action(self, action: Tuple[int, int, int, int]) -> bool:
        x, y, piece_index, rotation = action
        piece = np.rot90(self.pieces.pieces[piece_index], rotation)
        piece_width, piece_height = piece.shape

        # Verify if piece can be physically placed
        if np.any(self.board[x : x + piece_width, y : y + piece_height] != 0):
            return False

        # Verify if it touches one of the player's corners
        if not self.touches_own_corner(x, y, piece):
            return False

        # Verify if it's the first piece and if it's on the corner
        if self.n_turns < 1 and not self.is_on_corner(x, y, piece):
            return False

        # Verify if it touches own side
        if self.touches_own_side(x, y, piece):
            return False

        return True

    def touches_own_corner(self, x: int, y: int, piece: np.array) -> bool:
        piece_width, piece_height = piece.shape
        # Corners of the piece are ((0, 0), (0, piece_height-1), (piece_width-1, 0), (piece_width-1, piece_height-1))
        # Relative positions to the corners to check are ((-1, -1), (-1, 1), (1, -1), (1, 1))
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            for corner_x, corner_y in [
                (0, 0),
                (0, piece_height - 1),
                (piece_width - 1, 0),
                (piece_width - 1, piece_height - 1),
            ]:
                if (
                    self.is_empty_and_inside(x + corner_x + dx, y + corner_y + dy)
                    and self.board[x + corner_x + dx, y + corner_y + dy] == self.player
                ):
                    return True
        return False

    def is_on_corner(self, x: int, y: int, piece: np.array) -> bool:
        piece_width, piece_height = piece.shape
        # Check if any corner of the piece coincides with any of the board's corners
        for corner_x, corner_y in [
            (0, 0),
            (0, BOARD_SIZE - 1),
            (BOARD_SIZE - 1, 0),
            (BOARD_SIZE - 1, BOARD_SIZE - 1),
        ]:
            if (corner_x, corner_y) in [
                (x + dx, y + dy)
                for dx, dy in [
                    (0, 0),
                    (0, piece_height - 1),
                    (piece_width - 1, 0),
                    (piece_width - 1, piece_height - 1),
                ]
            ]:
                return True
        return False

    def is_empty_and_inside(self, x: int, y: int) -> bool:
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and self.board[x, y] == 0

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

    def evaluate_move(self, move):
        # Prefer placing larger pieces first
        _, _, piece_index, _ = move
        return np.sum(
            self.pieces.pieces[piece_index]
        )  # This works because piece cells are represented by 1's

    def visualize(self) -> str:
        return "\n".join(" ".join(str(int(cell)) for cell in row) for row in self.board)

    def __repr__(self) -> str:
        return "blokus"

    def touches_own_side(self, x: int, y: int, piece: np.array) -> bool:
        piece_width, piece_height = piece.shape

        # Relative positions to the sides to check are ((-1, 0), (1, 0), (0, -1), (0, 1))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for side_x in range(x, x + piece_width):
                for side_y in range(y, y + piece_height):
                    if (
                        self.is_empty_and_inside(side_x + dx, side_y + dy)
                        and self.board[side_x + dx, side_y + dy] == self.player
                    ):
                        return True
        return False

    @property
    def transposition_table_size(self):
        return 2 ** (BOARD_SIZE)


import numpy as np


class BlokusPieces:
    pieces = [
        np.array([[1]]),  # monomino
        np.array([[1, 1]]),  # domino
        np.array([[1, 1, 1]]),  # tromino_I
        np.array([[1, 0], [1, 0], [1, 1]]),  # tromino_L
        np.array([[1, 1, 1, 1]]),  # tetromino_I
        np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]]),  # tetromino_L
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]]),  # tetromino_T
        np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]]),  # tetromino_S
        np.array([[1, 1], [1, 1]]),  # tetromino_O
        np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]]),  # pentomino_F
        np.array([[1, 1, 1, 1, 1]]),  # pentomino_I
        np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]]),  # pentomino_L
        np.array([[1, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 1]]),  # pentomino_N
        np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0]]),  # pentomino_P
        np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]]),  # pentomino_T
        np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]]),  # pentomino_U
        np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]]),  # pentomino_V
        np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]]),  # pentomino_W
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),  # pentomino_X
        np.array([[1, 0], [1, 0], [1, 1], [1, 0], [1, 0]]),  # pentomino_Y
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]]),  # pentomino_Z
    ]

    def __init__(self):
        self.available_pieces = {(player, i): 2 for player in [1, 2] for i in range(len(BlokusPieces.pieces))}
        self.pieces_count = [self._piece_count(player) for player in [1, 2]]

    def copy(self):
        return copy.deepcopy(self)

    def rotate_piece(self, piece):
        return np.rot90(piece)

    def play_piece(self, piece_index, player):
        key = (player, piece_index)
        if self.available_pieces[key] > 0:
            self.available_pieces[key] -= 1
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

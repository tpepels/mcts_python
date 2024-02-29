# cython: language_level=3
import array
import cython

from fastrand import pcg32randint as randint
import numpy as np

from cython.cimports.includes import normalize, GameState, win, loss, draw
from cython.cimports.games.minishogi import PIECES, PIECE_CHARS, MATERIAL, P2_OFFS, flat_moves, move_indices
from termcolor import colored

flat_moves = [
    -1,
    0,
    -1,
    -1,
    0,
    -1,
    1,
    -1,
    1,
    0,
    1,
    1,
    0,
    1,
    -1,
    1,
    -1,
    0,
    -1,
    -1,
    -1,
    1,
    0,
    -1,
    0,
    1,
    1,
    0,
    -1,
    0,
    -1,
    0,
    -1,
    -1,
    -1,
    1,
    0,
    -1,
    0,
    1,
    1,
    0,
    -1,
    -1,
    -1,
    1,
    -1,
    0,
    1,
    -1,
    1,
    1,
    -1,
    0,
    -1,
    -1,
    -1,
    1,
    0,
    -1,
    0,
    1,
    1,
    0,
    -1,
    0,
    0,
    -1,
    1,
    0,
    0,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    1,
    -1,
    -1,
    0,
    0,
    -1,
    1,
    0,
    0,
    1,
    1,
    0,
    1,
    1,
    0,
    1,
    -1,
    1,
    -1,
    0,
    -1,
    -1,
    0,
    -1,
    1,
    -1,
    1,
    0,
    1,
    1,
    1,
    -1,
    0,
    1,
    0,
    -1,
    -1,
    0,
    1,
    0,
    1,
    0,
    1,
    1,
    1,
    -1,
    0,
    1,
    0,
    -1,
    -1,
    0,
    1,
    1,
    1,
    -1,
    1,
    0,
    -1,
    1,
    -1,
    -1,
    1,
    0,
    1,
    1,
    1,
    -1,
    0,
    1,
    0,
    -1,
    -1,
    0,
    1,
    0,
    0,
    1,
    -1,
    0,
    0,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    0,
    0,
    1,
    -1,
    0,
    0,
    -1,
]

move_indices = [0, 0, 16, 28, 30, 42, 52, 64, 72, 80, 88, 96, 112, 124, 126, 138, 148, 160, 168, 176, 184, 192]


P2_OFFS = 10

PIECES = {
    "K": 1,
    "G": 2,  # Regular gold general (cannot be promoted)
    "P": 3,
    "+P": 4,  # An upgraded pawn is a gold general
    "S": 5,
    "+S": 6,  # A promoted silver general is a gold general
    "R": 7,
    "+R": 8,  # A promoted rook is a dragon king
    "B": 9,
    "+B": 10,  # A promoted bishop is a dragon horse
}

MATERIAL = [
    0,  # Empty square (0)
    0,  # King (1)
    6,  # Gold General (2)
    1,  # Pawn (3)
    7,  # Promoted Pawn (Gold General) (4)
    5,  # Silver General (5)
    6,  # Promoted Silver General (Gold General) (6)
    9,  # Rook (7)
    12,  # Promoted Rook (Dragon King) (8)
    8,  # Bishop (9)
    10,  # Promoted Bishop (Dragon Horse) (10)
    0,  # King (11)
    6,  # Gold General (12)
    1,  # Pawn (13)
    7,  # Promoted Pawn (Gold General) (14)
    5,  # Silver General (15)
    6,  # Promoted Silver General (Gold General) (16)
    9,  # Rook (17)
    12,  # Promoted Rook (Dragon King) (18)
    8,  # Bishop (19)
    10,  # Promoted Bishop (Dragon Horse) (20)
]


# Define the pieces for both players, including promotions
PIECE_CHARS = list(PIECES.keys())


@cython.cclass
class MiniShogi(GameState):
    zobrist_table = [[[randint(1, 2**61 - 1) for _ in range(21)] for _ in range(5)] for _ in range(5)]

    REUSE = True

    board = cython.declare(cython.int[:, :], visibility="public")
    board_hash = cython.declare(cython.longlong, visibility="public")
    last_action = cython.declare(cython.tuple[cython.int, cython.int, cython.int, cython.int], visibility="public")
    current_player_moves = cython.declare(cython.list, visibility="public")

    captured_pieces_1 = cython.declare(cython.list[cython.int], visibility="public")
    captured_pieces_2 = cython.declare(cython.list[cython.int], visibility="public")

    king_1 = cython.declare(cython.tuple[cython.int, cython.int], visibility="public")
    king_2 = cython.declare(cython.tuple[cython.int, cython.int], visibility="public")

    # Private fields
    check = cython.declare(cython.bint, visibility="public")
    winner: cython.int
    state_occurrences: cython.dict

    def __init__(
        self,
        board=None,
        player=1,
        board_hash=0,
        last_action=(0, 0),
        captured_pieces_1=None,
        captured_pieces_2=None,
        king_1=(),
        king_2=(),
        state_occurences=None,
    ):
        # This needs to be determined after each move made.
        self.check = False
        self.winner = -1

        if board is None:
            self.board = self._init_board()
            self.board_hash = self._init_hash()

            # Keep track of the pieces on the board, this keeps evaluation and move generation from recomputing these over and over
            self.last_action = (-1, -1, -1, -1)
            self.player = 1

            self.captured_pieces_1 = []
            self.captured_pieces_2 = []
            self.current_player_moves = []
            self.state_occurrences = {}
        else:
            self.board = board
            self.player = player
            self.board_hash = board_hash
            self.last_action = last_action
            self.captured_pieces_1 = captured_pieces_1
            self.captured_pieces_2 = captured_pieces_2
            self.king_1 = king_1
            self.king_2 = king_2
            self.state_occurrences = state_occurences

    @cython.ccall
    def _init_board(self) -> cython.int[:, :]:
        board: cython.int[:, :] = np.zeros((5, 5), dtype=np.int32)

        # Set up the pieces for White (Player 2, top of the board)
        board[0, 0] = PIECES["K"] + P2_OFFS  # King
        board[0, 1] = PIECES["G"] + P2_OFFS  # Gold General
        board[0, 2] = PIECES["S"] + P2_OFFS  # Silver General
        board[0, 3] = PIECES["B"] + P2_OFFS  # Bishop
        board[0, 4] = PIECES["R"] + P2_OFFS  # Rook
        board[1, 0] = PIECES["P"] + P2_OFFS  # Pawn in front of the king

        # Set up the pieces for Black (Player 1, bottom of the board)
        board[4, 0] = PIECES["R"]  # Rook
        board[4, 1] = PIECES["B"]  # Bishop
        board[4, 2] = PIECES["S"]  # Silver General
        board[4, 3] = PIECES["G"]  # Gold General
        board[4, 4] = PIECES["K"]  # King
        board[3, 4] = PIECES["P"]  # Pawn in front of the king

        self.king_1 = (4, 4)  # Black's king
        self.king_2 = (0, 0)  # White's king

        return board

    @cython.ccall
    def _init_hash(self):
        # When a promoted piece is captured, it loses its promoted status.
        # If that piece is later dropped back onto the board by the player who captured it,
        # it returns as an unpromoted piece.
        # This means that we do not have to keep track of the promoted status of captured pieces.
        hash_value = 0
        for row in range(5):
            for col in range(5):
                piece = self.board[row, col]
                # Use the piece value directly as the index in the Zobrist table
                hash_value ^= MiniShogi.zobrist_table[row][col][piece]

        return hash_value

    @cython.ccall
    def skip_turn(self) -> GameState:
        """Used for the null-move heuristic in alpha-beta search"""
        # Pass the same board hash since this is only used for null moves
        return MiniShogi(
            board=self.board.copy(),
            player=self.player,
            board_hash=self.board_hash,
            last_action=self.last_action,
            captured_pieces_1=self.captured_pieces_1.copy(),
            captured_pieces_2=self.captured_pieces_2.copy(),
            king_1=self.king_1,
            king_2=self.king_2,
            state_occurences=self.state_occurrences.copy(),
        )

    @cython.cfunc
    @cython.locals(
        from_row=cython.int,
        from_col=cython.int,
        to_row=cython.int,
        to_col=cython.int,
        piece=cython.int,
        captured_piece=cython.int,
        promoted_piece=cython.int,
    )
    def apply_action_playout(self, action: cython.tuple) -> cython.void:
        if action is None:
            # This is a pass action, i.e. we have no moves to make and lost the game
            self.winner = 3 - self.player
            return

        from_row = action[0]
        from_col = action[1]
        to_row = action[2]
        to_col = action[3]

        if from_row == -1:
            # Handle drop action
            # ! This assumes that pieces are demoted on capture, i.e. the capture lists have only unpromoted pieces
            piece = from_col
            self.board[to_row, to_col] = piece
            # Remove the piece from the captured pieces list
            if self.player == 1:
                self.captured_pieces_1.remove(piece)
            else:
                self.captured_pieces_2.remove(piece)

        elif from_row == to_row and from_col == to_col:
            # Promotion action
            promoted_piece = self.get_promoted_piece(from_row, from_col)
            self.board[from_row, from_col] = promoted_piece
        else:
            # Handle move action
            piece = self.board[from_row, from_col]
            captured_piece = self.board[to_row, to_col]

            assert captured_piece != 11 and captured_piece != 1, "King capture \n" + self.visualize(True)

            if captured_piece != 0:
                # Demote the piece if it is captured (after updating the hash)
                captured_piece = self.get_demoted_piece(captured_piece)
                # Put the piece in the opposing player's captured pieces list, so they can drop them later
                if self.player == 1:
                    # Flip the piece to the opposing player's perspective
                    self.captured_pieces_1.append(captured_piece - P2_OFFS)
                else:
                    # Flip the piece to the opposing player's perspective
                    self.captured_pieces_2.append(captured_piece + P2_OFFS)

            # Update the king's position if a king is moved
            if piece == 1:  # Check if player 1's king is moved
                self.king_1 = (to_row, to_col)
            elif piece == 11:  # Check if player 2's king is moved
                self.king_2 = (to_row, to_col)

            # This will be the promoted piece if the move is a promotion, otherwise it will be the original piece
            self.board[to_row, to_col] = piece
            self.board[from_row, from_col] = 0

        # Update the last action
        self.last_action = action
        # Switch player after the action
        self.player = 3 - self.player
        # After each move, determine if the player to move is in check
        self.check = self.is_king_attacked(self.player)

        # self.winner = -2  # Reset the winner
        # # If the player is in check, check for checkmate
        # if self.check:
        #     if self.is_checkmate(self.player):
        #         self.winner = 3 - self.player
        #     else:
        #         # The -1 indicates that the game is not over, it prevents another check for checkmate
        #         self.winner = -1

    @cython.ccall
    @cython.locals(
        from_row=cython.int,
        from_col=cython.int,
        to_row=cython.int,
        to_col=cython.int,
        piece=cython.int,
        captured_piece=cython.int,
        promoted_piece=cython.int,
        new_state=MiniShogi,
    )
    def apply_action(self, action: cython.tuple) -> MiniShogi:
        # Create a copy of the current game state
        new_state = MiniShogi(
            board=self.board.copy(),
            player=self.player,
            board_hash=self.board_hash,
            last_action=self.last_action,
            captured_pieces_1=self.captured_pieces_1.copy(),
            captured_pieces_2=self.captured_pieces_2.copy(),
            king_1=self.king_1,
            king_2=self.king_2,
            state_occurences=self.state_occurrences.copy(),
        )
        if action is None:
            # This is a pass action, i.e. we have no moves to make and lost the game
            new_state.winner = 3 - new_state.player
            return new_state

        from_row = action[0]
        from_col = action[1]
        to_row = action[2]
        to_col = action[3]

        if from_row == -1:
            # Handle drop action
            # ! This assumes that pieces are demoted on capture, i.e. the capture lists have only unpromoted pieces
            piece = from_col
            new_state.board[to_row, to_col] = piece
            new_state.board_hash ^= MiniShogi.zobrist_table[to_row][to_col][piece]

            # Remove the piece from the captured pieces list
            if new_state.player == 1:
                new_state.captured_pieces_1.remove(piece)
            else:
                new_state.captured_pieces_2.remove(piece)

        elif from_row == to_row and from_col == to_col:

            piece = new_state.board[from_row, from_col]
            promoted_piece = new_state.get_promoted_piece(from_row, from_col)

            new_state.board_hash ^= MiniShogi.zobrist_table[from_row][from_col][piece]
            new_state.board_hash ^= MiniShogi.zobrist_table[from_row][from_col][promoted_piece]
            new_state.board[from_row, from_col] = promoted_piece

        else:
            # Handle move action
            piece = new_state.board[from_row, from_col]
            captured_piece = new_state.board[to_row, to_col]

            new_state.board_hash ^= MiniShogi.zobrist_table[from_row][from_col][piece]
            new_state.board_hash ^= MiniShogi.zobrist_table[to_row][to_col][piece]

            if captured_piece != 0:
                new_state.board_hash ^= MiniShogi.zobrist_table[to_row][to_col][captured_piece]
                # Demote the piece if it is captured (after updating the hash)
                captured_piece = new_state.get_demoted_piece(captured_piece)
                # Put the piece in the opposing player's captured pieces list, so they can drop them later
                if new_state.player == 1:
                    # Flip the piece to the opposing player's perspective
                    new_state.captured_pieces_1.append(captured_piece - P2_OFFS)
                else:
                    # Flip the piece to the opposing player's perspective
                    new_state.captured_pieces_2.append(captured_piece + P2_OFFS)

            # Update the king's position if a king is moved
            if piece == PIECES["K"]:  # Check if player 1's king is moved
                new_state.king_1 = (to_row, to_col)
            elif piece == PIECES["K"] + P2_OFFS:  # Check if player 2's king is moved
                new_state.king_2 = (to_row, to_col)

            # This will be the promoted piece if the move is a promotion, otherwise it will be the original piece
            new_state.board[to_row, to_col] = piece
            new_state.board[from_row, from_col] = 0

        # Update the last action
        new_state.last_action = action

        # Switch player after the action
        new_state.player = 3 - new_state.player
        # After each move, determine if I am in check
        new_state.check = new_state.is_king_attacked(new_state.player)

        # new_state.winner = -2  # Reset the winner
        if new_state.check:
            # Increment the count for the current board hash, initializing it to 0 if it doesn't exist.
            new_state.state_occurrences[new_state.board_hash] = (
                new_state.state_occurrences.get(new_state.board_hash, 0) + 1
            )
            # Check for repetition loss
            if new_state.state_occurrences[new_state.board_hash] == 4:
                # The player repeating check causing a position to repeat 4 times in a row is a loss
                new_state.winner = 3 - new_state.player
            # else:
            # # Check for checkmate
            # if new_state.is_checkmate(new_state.player):
            #     new_state.winner = 3 - new_state.player
            # else:
            #     # The -1 indicates that the game is not over, it prevents another check for checkmate
            #     new_state.winner = -1

        return new_state

    @cython.cfunc
    @cython.locals(
        row=cython.int,
        col=cython.int,
        piece=cython.int,
        player_piece_start=cython.int,
        player_piece_end=cython.int,
        from_row=cython.int,
        from_col=cython.int,
        to_row=cython.int,
        to_col=cython.int,
        is_defense=cython.bint,
        i=cython.int,
        dx=cython.int,
        dy=cython.int,
        idx=cython.int,
        base_piece=cython.int,
        new_row=cython.int,
        new_col=cython.int,
        start_index=cython.int,
    )
    def get_random_action(self) -> cython.tuple:
        assert self.winner <= -1, "Cannot get legal actions for a terminal state."

        moves: cython.list = []
        player_piece_start, player_piece_end = (1, 10) if self.player == 1 else (11, 20)

        # Handle move and promotion actions
        for row in range(5):
            for col in range(5):
                piece = self.board[row, col]
                # Piece belongs to the current player
                if player_piece_start <= piece <= player_piece_end:
                    # Check if the move is a promotion, promotion can never break check
                    if not self.check and self.is_promotion(row, piece):
                        move: cython.tuple[cython.int, cython.int, cython.int, cython.int] = (row, col, row, col)
                        moves.append(move)
                        moves.append(move)
                        moves.append(move)
                        moves.append(move)
                        moves.append(move)
                        moves.append(move)

                    # Pawn on the last rows cannot move
                    if not (piece == 3 and row == 0) and not (piece == 13 and row == 4):
                        # Here, don't use _generate_moves, such that we increase performance
                        if 7 <= piece <= 10 or 17 <= piece <= 20:
                            base_piece = 7 if piece == 7 or piece == 8 or piece == 17 or piece == 18 else 9
                            if piece >= 11:
                                base_piece += P2_OFFS

                                move_count: cython.int = (move_indices[base_piece + 1] - move_indices[base_piece]) // 2
                                start_random_idx: cython.int = randint(0, move_count - 1)

                                for i in range(move_count):
                                    # Circular iteration using modulo
                                    idx = move_indices[base_piece] + ((start_random_idx + i) % move_count) * 2
                                    new_row, new_col = row, col

                                    while 1:
                                        new_row += flat_moves[idx]
                                        new_col += flat_moves[idx + 1]

                                        if not (0 <= new_row < 5 and 0 <= new_col < 5) or (
                                            player_piece_start <= self.board[new_row, new_col] <= player_piece_end
                                        ):
                                            break  # Out of bounds or own piece encountered

                                        if not self.simulate_move_exposes_king(row, col, new_row, new_col):
                                            move: cython.tuple[cython.int, cython.int, cython.int, cython.int] = (
                                                row,
                                                col,
                                                new_row,
                                                new_col,
                                            )
                                            if self.board[new_row, new_col] != 0:
                                                moves.append(move)
                                                moves.append(move)
                                                moves.append(move)
                                            else:
                                                moves.append(move)

                                        if self.board[new_row, new_col] != 0:
                                            break  # Opposing piece encountered, stop extending

                            if piece != 7 and piece != 9 and piece != 17 and piece != 19:
                                # Logic for standard and limited one-square moves

                                move_count: cython.int = (move_indices[piece + 1] - move_indices[piece]) // 2
                                start_random_idx: cython.int = randint(0, move_count - 1)

                                for i in range(move_count):
                                    idx = move_indices[piece] + ((start_random_idx + i) % move_count) * 2
                                    new_row = row + flat_moves[idx]
                                    new_col = col + flat_moves[idx + 1]

                                    if not (0 <= new_row < 5 and 0 <= new_col < 5) or (
                                        player_piece_start <= self.board[new_row, new_col] <= player_piece_end
                                    ):
                                        continue  # Out of bounds or own piece encountered

                                    if not self.simulate_move_exposes_king(row, col, new_row, new_col):
                                        move: cython.tuple[cython.int, cython.int, cython.int, cython.int] = (
                                            row,
                                            col,
                                            new_row,
                                            new_col,
                                        )
                                        if self.board[new_row, new_col] != 0:
                                            moves.append(move)
                                            moves.append(move)
                                            moves.append(move)
                                        else:
                                            moves.append(move)

        # Handle drop actions
        captured_pieces: cython.list[cython.int] = (
            self.captured_pieces_1 if self.player == 1 else self.captured_pieces_2
        )
        for i in range(len(captured_pieces)):
            piece = captured_pieces[i]
            for row in range(5):
                for col in range(5):
                    if self.board[row, col] == 0:  # Empty square
                        if self.is_legal_drop(row, col, piece, self.player):
                            move: cython.tuple[cython.int, cython.int, cython.int, cython.int] = (
                                -1,
                                piece,
                                row,
                                col,
                            )
                            if self.check:
                                if not self.simulate_move_exposes_king(-1, piece, row, col):
                                    moves.append(move)
                                    moves.append(move)
                                    moves.append(move)
                                    moves.append(move)
                                    moves.append(move)
                            else:
                                moves.append(move)
                                moves.append(move)
                                moves.append(move)
                                moves.append(move)

        if len(moves) == 0:
            self.winner = 3 - self.player
            return None

        # assert len(moves) > 0, "No legal actions found for player " + str(self.player) + "\n" + self.visualize(True)
        return moves[randint(0, len(moves) - 1)]

    @cython.ccall
    @cython.locals(
        row=cython.int, col=cython.int, piece=cython.int, player_piece_start=cython.int, player_piece_end=cython.int
    )
    def get_legal_actions(self) -> cython.list:
        assert self.winner <= -1, "Cannot get legal actions for a terminal state.\n" + self.visualize(False)

        legal_actions = []
        player_piece_start, player_piece_end = (1, 10) if self.player == 1 else (11, 20)

        # Handle move and promotion actions
        for row in range(5):
            for col in range(5):
                piece = self.board[row, col]
                # Piece belongs to the current player
                if player_piece_start <= piece <= player_piece_end:
                    # Check if the move is a promotion, promotion can never break check
                    if not self.check and self.is_promotion(row, piece):
                        legal_actions.append((row, col, row, col))

                    # Pawn on the last rows cannot move (this makes sure not to generate moves that are not needed)
                    if not (piece == 3 and row == 0) and not (piece == 13 and row == 4):
                        # Generate moves for this piece
                        self.get_moves(row, col, legal_actions)

        # Handle drop actions
        captured_pieces = self.captured_pieces_1 if self.player == 1 else self.captured_pieces_2

        for piece in captured_pieces:
            for row in range(5):
                for col in range(5):
                    if self.board[row, col] == 0:  # Empty square
                        if self.is_legal_drop(row, col, piece, self.player):
                            if self.check:
                                # We have to determine whether this move relieves the check
                                if not self.simulate_move_exposes_king(-1, piece, row, col):
                                    legal_actions.append((-1, piece, row, col))
                            else:
                                # We only have to check if the drop exposes the king to an attack if we are in check
                                legal_actions.append((-1, piece, row, col))

        # A None action means a kind of pass that signifies that the player has no legal moves
        if len(legal_actions) == 0:
            self.winner = 3 - self.player
            return [None]

        return legal_actions

    @cython.cfunc
    @cython.locals(is_drop=cython.bint, original_piece=cython.int, moving_piece=cython.int, exposes_king=cython.bint)
    def simulate_move_exposes_king(
        self, from_row: cython.int, from_col: cython.int, to_row: cython.int, to_col: cython.int
    ) -> cython.bint:
        """
        Simulates a move or a drop action and checks if it exposes the king to an attack.

        Args:
        from_row (cython.int): The row from which a piece is moved; -1 if it's a drop.
        from_col (cython.int): The column from which a piece is moved; -1 if it's a drop.
        to_row (cython.int): The target row for the move or drop.
        to_col (cython.int): The target column for the move or drop.
        piece (cython.int, optional): The piece to be dropped; only used for drop actions.

        Returns:
        bool: True if the action exposes the king to an attack, False otherwise.
        """
        # Determine if this is a move or a drop
        is_drop = from_row == -1

        # Store the original state
        original_piece = self.board[to_row, to_col]
        moving_piece = from_col if is_drop else self.board[from_row, from_col]

        # Perform the action (move or drop)
        self.board[to_row, to_col] = moving_piece
        if not is_drop:
            self.board[from_row, from_col] = 0

        # Check if the king is attacked after this action
        player = 1 if moving_piece <= P2_OFFS else 2
        if moving_piece == 1:
            self.king_1 = (to_row, to_col)
        elif moving_piece == 11:
            self.king_2 = (to_row, to_col)

        exposes_king = self.is_king_attacked(player)

        # Restore the original state
        self.board[to_row, to_col] = original_piece

        if moving_piece == 1:
            self.king_1 = (from_row, from_col)
        elif moving_piece == 11:
            self.king_2 = (from_row, from_col)

        if not is_drop:
            self.board[from_row, from_col] = moving_piece

        return exposes_king

    @cython.cfunc
    def is_legal_drop(self, row: cython.int, col: cython.int, piece: cython.int, player: cython.int) -> cython.bint:
        # Check if the target square is empty
        if self.board[row, col] != 0:
            return False  # We can only drop on unoccupied squares
        captured_pieces: cython.list = self.captured_pieces_1 if player == 1 else self.captured_pieces_2
        # Pawn-specific rules
        if piece == 3 or piece == 13:
            # Check for another unpromoted pawn in the same column and last row restriction
            for r in range(5):
                if self.board[r, col] == piece or (player == 1 and row == 0) or (player == 2 and row == 4):
                    return False

            # Temporarily drop the piece
            self.board[row, col] = piece
            captured_pieces.remove(piece)
            # Check if this results in checkmate
            checkmate = self.is_checkmate(3 - player)
            captured_pieces.append(piece)
            # Undo the drop
            self.board[row, col] = 0
            # Dropping a pawn may not result in checkmate
            return not checkmate
        # In any other case, a piece can move
        return True

    @cython.cfunc
    @cython.locals(row=cython.int, col=cython.int, piece=cython.int, i=cython.int)
    def is_checkmate(self, player: cython.int) -> cython.bint:
        """
        Check if the given player is in checkmate. This means that the player to move, cannot make any move.

        Args:
        player (cython.int): The player to check for checkmate.

        Returns:
        cython.bint: 1 if the player is in checkmate, otherwise 0.
        """
        # ! The check check is done after every move. So no reason to repeat checkmate check again.
        if not self.check:
            return 0

        # Check for any legal move by the pieces
        for row in range(5):
            for col in range(5):
                piece = self.board[row, col]
                if (player == 1 and 1 <= piece <= 10) or (player == 2 and 11 <= piece <= 20):
                    if self.has_legal_move(row, col, piece, player):
                        return 0  # Found a legal move

        # Check for any legal drop that can resolve the check
        captured_pieces: cython.list[cython.int] = self.captured_pieces_1 if player == 1 else self.captured_pieces_2
        for i in range(len(captured_pieces)):
            piece = captured_pieces[i]
            for row in range(5):
                for col in range(5):
                    if self.is_legal_drop(row, col, piece, player):
                        # We have to determine whether this move relieves the check
                        if not self.simulate_move_exposes_king(-1, piece, row, col):
                            return 0
        return 1  # No legal moves or drops, so it's checkmate

    @cython.cfunc
    @cython.locals(
        row=cython.int,
        col=cython.int,
        dx=cython.int,
        dy=cython.int,
        idx=cython.int,
        piece=cython.int,
        player_piece_start=cython.int,
        player_piece_end=cython.int,
        base_piece=cython.int,
        new_row=cython.int,
        new_col=cython.int,
        i=cython.int,
        start_index=cython.int,
    )
    def _generate_moves(
        self,
        row: cython.int,
        col: cython.int,
        piece: cython.int,
        callback,
        randomize: cython.bint = 0,
        count_defense: cython.bint = 0,
    ):
        player_piece_start, player_piece_end = (1, 10) if piece <= P2_OFFS else (11, 20)

        if 7 <= piece <= 10 or 17 <= piece <= 20:
            base_piece = 7 if piece == 7 or piece == 8 or piece == 17 or piece == 18 else 9
            if piece >= 11:
                base_piece += P2_OFFS

            move_count: cython.int = (move_indices[base_piece + 1] - move_indices[base_piece]) // 2
            start_random_idx: cython.int = randint(0, move_count - 1) if randomize else 0

            for i in range(move_count):
                # Circular iteration using modulo
                idx = move_indices[base_piece] + ((start_random_idx + i) % move_count) * 2
                new_row, new_col = row, col

                while 1:
                    new_row += flat_moves[idx]
                    new_col += flat_moves[idx + 1]

                    if (
                        count_defense
                        and (0 <= new_row < 5 and 0 <= new_col < 5)
                        and player_piece_start <= self.board[new_row, new_col] <= player_piece_end
                    ):
                        # If defensive move, call the callback with an additional flag
                        callback(row, col, new_row, new_col, piece, True)

                    if not (0 <= new_row < 5 and 0 <= new_col < 5) or (
                        player_piece_start <= self.board[new_row, new_col] <= player_piece_end
                    ):
                        break  # Out of bounds or own piece encountered

                    if callback(row, col, new_row, new_col, piece, False):
                        return 1  # Callback indicates to stop extending

                    if self.board[new_row, new_col] != 0:
                        break  # Opposing piece encountered, stop extending

        if piece != 7 and piece != 9 and piece != 17 and piece != 19:
            # Logic for standard and limited one-square moves
            move_count: cython.int = (move_indices[piece + 1] - move_indices[piece]) // 2
            start_random_idx: cython.int = randint(0, move_count - 1) if randomize else 0

            for i in range(move_count):
                idx = move_indices[piece] + ((start_random_idx + i) % move_count) * 2
                new_row = row + flat_moves[idx]
                new_col = col + flat_moves[idx + 1]

                if (
                    count_defense
                    and (0 <= new_row < 5 and 0 <= new_col < 5)
                    and player_piece_start <= self.board[new_row, new_col] <= player_piece_end
                ):
                    # If defensive move, call the callback with an additional flag
                    callback(row, col, new_row, new_col, piece, True)

                # Out of bounds or own piece
                if not (0 <= new_row < 5 and 0 <= new_col < 5) or (
                    player_piece_start <= self.board[new_row, new_col] <= player_piece_end
                ):
                    continue  # Out of bounds or own piece encountered

                if callback(row, col, new_row, new_col, piece, False):
                    return 1  # Callback indicates to stop extending

        return 0  # No valid move processed or no need to stop extending

    @cython.cfunc
    def get_moves(self, row: cython.int, col: cython.int, moves: cython.list) -> cython.list:
        def move_callback(
            from_row: cython.int,
            from_col: cython.int,
            to_row: cython.int,
            to_col: cython.int,
            piece: cython.int,
            is_defense: cython.bint,
        ):
            if not self.simulate_move_exposes_king(from_row, from_col, to_row, to_col):
                moves.append((from_row, from_col, to_row, to_col))
            return False  # Continue generating all moves

        self._generate_moves(
            row,
            col,
            self.board[row, col],
            callback=move_callback,
            randomize=True,
            count_defense=False,
        )
        return moves

    @cython.cfunc
    @cython.locals(
        opposing_player_piece_start=cython.int,
        opposing_player_piece_end=cython.int,
        row=cython.int,
        col=cython.int,
        piece=cython.int,
    )
    def is_king_attacked(self, player: cython.int) -> cython.bint:
        king_pos: cython.tuple[cython.int, cython.int] = self.king_1 if player == 1 else self.king_2

        def move_callback(
            from_row: cython.int,
            from_col: cython.int,
            to_row: cython.int,
            to_col: cython.int,
            piece: cython.int,
            is_defense: cython.bint,
        ):
            return (to_row, to_col) == king_pos  # Return True if king is attacked, this stops _generate_moves

        opposing_player_piece_start, opposing_player_piece_end = (11, 20) if player == 1 else (1, 10)
        for row in range(5):
            for col in range(5):
                piece = self.board[row, col]
                if opposing_player_piece_start <= piece <= opposing_player_piece_end:
                    if self._generate_moves(
                        row,
                        col,
                        piece,
                        callback=move_callback,
                        randomize=True,
                        count_defense=False,
                    ):
                        return True  # King is attacked
        return False

    @cython.cfunc
    def has_legal_move(self, row: cython.int, col: cython.int, piece: cython.int, player: cython.int) -> cython.bint:
        def move_callback(
            from_row: cython.int,
            from_col: cython.int,
            to_row: cython.int,
            to_col: cython.int,
            piece: cython.int,
            is_defense: cython.bint,
        ):
            # Return True if move is legal, this stops _generate_moves
            return not self.simulate_move_exposes_king(from_row, from_col, to_row, to_col)

        return self._generate_moves(
            row,
            col,
            piece,
            callback=move_callback,
            count_defense=False,
            randomize=True,
        )

    @cython.ccall
    def is_terminal(self) -> cython.bint:
        """
        Check if the current game state is terminal (i.e., a player has won).

        :return: True if the game state is terminal, False otherwise.
        """
        # Any number higher than -1 means that we have found a winner.
        if self.winner >= 0:
            return 1
        else:
            return 0
        # # -1 means that we did not yet check whether we are in a terminal state, so if we are in check, the state may be terminal
        # if (
        #     self.check and self.winner == -2
        # ):  # This means the game is in check, and the checkmate check wat not performed.
        #     assert False, "Not sure we should be here" + self.visualize()
        #     # if self.is_checkmate(self.player):
        #     #     self.winner = 3 - self.player
        #     #     return 1

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def get_reward(self, player: cython.int) -> cython.int:
        if self.winner == player:
            return win
        elif self.winner == 3 - player:
            return loss
        return draw

    @cython.ccall
    def get_result_tuple(self) -> cython.tuple:
        if self.winner == 1:
            return (1.0, 0.0)
        elif self.winner == 2:
            return (0.0, 1.0)

        return (0.5, 0.5)

    @cython.cfunc
    def is_capture(self, move: cython.tuple) -> cython.bint:
        """
        Check if the given move results in a capture.

        :param move: The move to check as a tuple (from_position, to_position).
        :return: True if the move results in a capture, False otherwise.
        """
        # Check if the destination cell contains an opponent's piece
        i: cython.int = move[2]
        j: cython.int = move[3]
        return self.board[i, j] == 3 - self.player

    # These dictionaries are used by run_games to set the parameter values
    param_order: dict = {
        "m_check": 0,
        "m_attacks": 1,
        "m_captures": 2,
        "m_material": 3,
        "m_dominance": 4,
        "m_defenses": 5,
        "a": 6,
    }

    default_params = array.array("d", [50.0, 1.0, 1.0, 3.0, 1.0, 0.5, 200.0])

    @cython.cfunc
    @cython.exceptval(-9999999, check=False)
    @cython.locals(
        from_row=cython.int,
        from_col=cython.int,
        to_row=cython.int,
        to_col=cython.int,
        piece=cython.int,
        is_defense=cython.bint,
        multip=cython.int,
        row=cython.int,
        col=cython.int,
        i=cython.int,
    )
    def evaluate(
        self,
        player: cython.int,
        params: cython.double[:],
        norm: cython.bint = 0,
    ) -> cython.double:
        # TODO hier was je gebleven, deze moet je nog goed nakijken... Er was bijvoorbeeld iets vreemds met de defense, die niet deterministisch was..
        attacks: cython.int = 0
        captures: cython.int = 0
        defenses: cython.int = 0
        board_control: cython.int = 0
        material_score: cython.int = 0
        score: cython.float = 0

        def callback(
            from_row: cython.int,
            from_col: cython.int,
            to_row: cython.int,
            to_col: cython.int,
            piece: cython.int,
            is_defense: cython.bint,
        ):
            nonlocal attacks, defenses, board_control, material_score
            multip: cython.int = (
                1 if (player == 1 and 1 <= piece <= P2_OFFS) or (player == 2 and 11 <= piece <= 20) else -1
            )

            if self.board[to_row, to_col] != 0:
                if not is_defense:
                    attacks += multip * MATERIAL[self.board[to_row, to_col]]
                else:
                    # print(f"{piece} is defending {self.board[to_row, to_col]} at {to_row},{to_col}")
                    defenses += multip * MATERIAL[self.board[to_row, to_col]]

            elif not is_defense:
                board_control += multip

            return False

        for row in range(5):
            for col in range(5):
                if self.board[row, col] != 0:
                    piece: cython.int = self.board[row, col]
                    multip: cython.int = (
                        1 if (player == 1 and 1 <= piece <= P2_OFFS) or (player == 2 and 11 <= piece <= 20) else -1
                    )
                    material_score += multip * MATERIAL[piece]
                    self._generate_moves(row, col, self.board[row, col], callback, count_defense=True, randomize=False)

        captured_pieces_player: cython.list[cython.int] = (
            self.captured_pieces_1 if player == 1 else self.captured_pieces_2
        )
        captured_pieces_opponent: cython.list[cython.int] = (
            self.captured_pieces_2 if player == 1 else self.captured_pieces_1
        )

        # Accumulate captures for the current player
        for i in range(len(captured_pieces_player)):
            captures += MATERIAL[captured_pieces_player[i]]

        # Subtract captures for the opponent
        for i in range(len(captured_pieces_opponent)):
            captures -= MATERIAL[captured_pieces_opponent[i]]

        if self.check and player == self.player:
            score = -params[0]
        elif self.check and player != self.player:
            score = params[0]

        score += (
            attacks * params[1]
            + captures * params[2]
            + material_score * params[3]
            + board_control * params[4]
            + defenses * params[5]
        )
        # print(
        #     f"Final Score Components - Attacks: {attacks}, Captures: {captures}, Material Score: {material_score}, Board Control: {board_control}, Defenses: {defenses}, Score: {score}"
        # )

        if norm:
            normalized_score = normalize(score, params[6])
            return normalized_score
        else:
            return score

    @cython.cfunc
    def evaluate_moves(self, moves: cython.list) -> cython.list:
        """
        Evaluate the given moves using a simple heuristic: each step forward is worth 1 point,
        and capturing an opponent's piece is worth 2 points.

        :param moves: The list of moves to evaluate.
        :return: The list of heuristic values of the moves.
        """
        scores: cython.list = [()] * len(moves)
        i: cython.int
        for i in range(len(moves)):
            scores[i] = (moves[i], self.evaluate_move(moves[i]))
        return scores

    @cython.cfunc
    def move_weights(self, moves: cython.list) -> cython.list:
        """
        :param moves: The list of moves to evaluate.
        :return: A list of scores for each move.
        """
        scores = [0] * len(moves)  # TODO In de c++ games gebruik je hier een vector
        for i in range(len(moves)):
            scores[i] = self.evaluate_move(moves[i])
        return scores

    @cython.ccall
    @cython.exceptval(-1, check=False)
    @cython.locals(
        from_row=cython.int,
        from_col=cython.int,
        to_row=cython.int,
        to_col=cython.int,
        moving_piece=cython.int,
        score=cython.int,
        def_row=cython.int,
        def_col=cython.int,
        i=cython.int,
    )
    def evaluate_move(self, move: cython.tuple) -> cython.int:
        """
        Evaluate the given move using a simple heuristic for move ordering

        :param move: The move to evaluate.
        :return: The heuristic value of the move.
        """
        from_row = move[0]
        from_col = move[1]
        to_row = move[2]
        to_col = move[3]
        moving_piece = from_col if from_row == -1 else self.board[from_row, from_col]

        score = 0

        # Check for capture and use MATERIAL value for the captured piece
        if self.board[to_row, to_col] != 0:
            score += MATERIAL[self.board[to_row, to_col]]

        # Promotions generally gain material advantage
        if from_row == to_row and from_col == to_col:
            score += MATERIAL[self.get_promoted_piece(to_row, to_col)]  # Promotion value

        # Drops (from_row == -1) are generally strong moves in MiniShogi
        if from_row == -1:
            score += MATERIAL[moving_piece]

        # Evaluate defensive or offensive nature of the move
        # * This is not perfect for rooks and bishops and promoting pieces, but it's good enough for now
        move_count: cython.int = (move_indices[moving_piece + 1] - move_indices[moving_piece]) // 2

        # Assuming there's no randomness needed here, so we'll iterate in order
        for i in range(move_count):
            # Calculate the index into flat_moves
            idx = move_indices[moving_piece] + i * 2  # Skipping by 2 because moves are in (dx, dy) pairs

            # Apply the move deltas to the to_row and to_col
            def_row = to_row + flat_moves[idx]
            def_col = to_col + flat_moves[idx + 1]

            if 0 <= def_row < 5 and 0 <= def_col < 5:
                if self.board[def_row, def_col] != 0:
                    if self.is_same_player_piece(moving_piece, self.board[def_row, def_col]):
                        score += MATERIAL[self.board[def_row, def_col]] / 2  # Defensive move
                    else:
                        score += MATERIAL[self.board[def_row, def_col]]  # Offensive move
                else:
                    score += 1  # The moves adds board dominance

        return cython.cast(cython.int, score)

    @cython.cfunc
    @cython.inline
    def is_same_player_piece(self, piece1: cython.int, piece2: cython.int) -> cython.bint:
        """
        Check if two pieces belong to the same player.

        :param piece1: First piece.
        :param piece2: Second piece.
        :return: True if both pieces belong to the same player, otherwise False.
        """
        return (piece1 <= P2_OFFS and piece2 <= P2_OFFS) or (piece1 > P2_OFFS and piece2 > P2_OFFS)

    @cython.ccall
    @cython.locals(i=cython.int, j=cython.int)
    def visualize(self, full_debug: cython.bint = False) -> cython.str:
        """
        Visualize the Minishogi board with additional debug information.

        :param full_debug: Whether to show full debug information (like board hash).
        :return: String representation of the board and additional information.
        """
        # First, invert the PIECES dictionary to map numbers to symbols
        NUM_TO_PIECES = {value: key for key, value in PIECES.items()}
        board_str = "Minishogi Board:\n"
        for i in range(5):
            for j in range(5):
                piece = self.board[i, j]

                char_index = piece % P2_OFFS - 1  # Adjust according to your piece encoding
                is_promoted = self.get_demoted_piece(piece) != piece
                is_player_2 = piece > P2_OFFS

                # Get the character for the piece and invert for Player 2 if needed
                char = PIECE_CHARS[char_index] if char_index >= 0 else "."
                if is_player_2:
                    char = get_upside_down_char(char)  # Assuming you have a function to do this

                # Use termcolor for differentiating piece states
                if is_promoted and is_player_2:
                    colored_char = colored(char, "blue")  # Promoted Player 2
                elif is_promoted:
                    colored_char = colored(char, "green")  # Promoted Player 1
                elif is_player_2:
                    colored_char = colored(char, "red")  # Normal Player 2
                else:
                    colored_char = colored(char, "cyan")  # Normal Player 1

                board_str += colored_char + " "
            board_str += "\n"

        # Additional debug information
        if full_debug:
            board_str += f"Board Hash: {self.board_hash}\n"
            board_str += f"Mate: {'Yes' if self.is_terminal() else 'No'} \n"
            board_str += f"Check: {'Yes' if self.check else 'No'}\n"
            board_str += f"Winner: {self.winner}\n"
            board_str += f"Player: {self.player}\n"
            self.check = True
            board_str += f"is_checkmate(self.player): {self.is_checkmate(self.player)}\n"
            self.check = False
            board_str += f"is_king_attacked(self.player): {self.is_king_attacked(self.player)}\n"
            board_str += f"King 1: {self.king_1} | King 2: {self.king_2}\n"

            # Then, convert the captured pieces lists for each player into their symbols
            captured_pieces_1_symbols = [NUM_TO_PIECES[piece] for piece in self.captured_pieces_1]
            captured_pieces_2_symbols = [NUM_TO_PIECES[piece - P2_OFFS] for piece in self.captured_pieces_2]

            # Finally, create strings from these lists to include in the board representation
            captured_1_str = ", ".join(captured_pieces_1_symbols)
            captured_2_str = ", ".join(captured_pieces_2_symbols)

            board_str += f"Captured Pieces Player 1: {captured_1_str}\n"
            board_str += f"Captured Pieces Player 2: {captured_2_str}\n"

            # Evaluations for each player
            p1_eval = self.evaluate(1, self.default_params)
            p2_eval = self.evaluate(2, self.default_params)
            board_str += f"Player 1 Evaluation: {p1_eval}\n"
            board_str += f"Player 2 Evaluation: {p2_eval}\n"

            # All legal actions
            # if not self.is_terminal():
            #     legal_actions = self.get_legal_actions()
            #     # Move evaluations (assuming you have a function for this)
            #     move_evaluations = [self.evaluate_move(move) for move in legal_actions]
            #     actions_evaluations_zip = zip(legal_actions, move_evaluations)
            #     # Formatting the string to include action and evaluation on each line
            #     board_str += "\n".join(
            #         [
            #             f"Action: {self.readable_move(action)}, Evaluation: {evaluation}"
            #             for action, evaluation in actions_evaluations_zip
            #         ]
            #     )

        return board_str

    @cython.ccall
    def readable_move(self, move):
        # Define mappings for columns and rows
        cols = ["a", "b", "c", "d", "e"]
        rows = ["1", "2", "3", "4", "5"]

        # Unpack the move
        from_row, from_col, to_row, to_col = move

        # Handle drop moves
        if from_row == -1:
            return f"Drop {from_col} to {cols[to_col]}{rows[to_row]}"

        # Convert indices to readable format for regular moves
        from_pos = cols[from_col] + rows[from_row]
        to_pos = cols[to_col] + rows[to_row]

        return f"{from_pos} to {to_pos}"

    @property
    def transposition_table_size(self) -> int:
        # return an appropriate size based on the game characteristics
        return 2**18

    def __repr__(self) -> str:
        game: str = "minishogi"
        return game

    @cython.cfunc
    @cython.inline
    def is_promotion(self, row: cython.int, piece: cython.int) -> cython.bint:
        if piece % 2 == 1:
            return (row == 0 and 2 < piece < 10) or (piece > 12 and row == 4)
        return False

    @cython.cfunc
    @cython.inline
    @cython.exceptval(-1, check=False)
    def get_promoted_piece(self, row: cython.int, col: cython.int) -> cython.int:
        piece: cython.int = self.board[row, col]
        # Promote if piece is unpromoted
        if piece % 2 == 1 and (2 < piece < 10) or piece > 12:
            return piece + 1
        return piece

    @cython.cfunc
    @cython.inline
    @cython.exceptval(-1, check=False)
    def get_demoted_piece(self, piece: cython.int) -> cython.int:
        # Even pieces > 3 are promoted (1 - king, 2 - Gold Gen. 3 - Pawn (4 - +Pawn))
        if piece % 2 == 0 and ((3 < piece <= 10) or piece > 13):
            return piece - 1

        return piece


def get_upside_down_char(char):
    # Mapping of regular characters to their upside-down Unicode equivalents
    upside_down_map = {
        "K": "ʞ",
        "p": "b",
        "P": "d",
        "s": "s",
        "S": "S",
        "r": "ɹ",
        "R": "ᴚ",
        "b": "q",
        "B": "ꓭ",
        "G": "⅁",
    }
    return upside_down_map.get(char, char)  # Return the upside-down char if exists, else return the char itself

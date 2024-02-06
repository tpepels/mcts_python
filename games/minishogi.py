# cython: language_level=3, wraparound=False, infer_types=True
import array
import cython

from fastrand import pcg32randint as randint
import numpy as np

from cython.cimports.includes import normalize, GameState, win, loss, draw, f_index
from cython.cimports.games.minishogi import MOVES, PIECES, PIECE_CHARS, MATERIAL, P2_OFFS
from termcolor import colored

# TODO Idee om move access te versnellen mocht het nodig zijn
# cimport cython
# from libc.stdlib cimport malloc, free

# # Example flattened move vectors for all pieces (simplified for illustration)
# cdef int[:] flat_moves = [1, 0, -1, 0, 0, 1, 0, -1, ...]  # Flat list of move deltas

# # Example start-end indices for each piece's moves in the flat_moves array
# cdef int[:] move_indices = [0, 4, 8, ...]  # Starting index for each piece's moves

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef get_moves_for_piece(int piece_id):
#     start_idx = move_indices[piece_id]
#     end_idx = move_indices[piece_id + 1]
#     # Access the moves for the specified piece using the start and end indices
#     for i in range(start_idx, end_idx, 2):
#         dx = flat_moves[i]
#         dy = flat_moves[i + 1]
#         # Process the move (dx, dy) for the piece


MOVES = [
    [],  # Empty square placeholder
    [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)],  # King (Player 1)
    [(-1, 0), (-1, -1), (-1, 1), (0, -1), (0, 1), (1, 0)],  # Gold General (Player 1, regular)
    [(-1, 0)],  # Pawn (Player 1)
    [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (0, 1)],  # Promoted Pawn (Player 1, Gold General)
    [(-1, -1), (-1, 1), (-1, 0), (1, -1), (1, 1)],  # Silver General (Player 1)
    [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (0, 1)],  # Promoted Silver General (Player 1, Gold General)
    [(-1, 0), (0, -1), (1, 0), (0, 1)],  # Rook (Player 1)
    [(-1, -1), (1, -1), (1, 1), (-1, 1)],  # Promoted Rook (Player 1, Dragon King)
    [(-1, -1), (1, 1), (-1, 1), (1, -1)],  # Bishop (Player 1)
    [(-1, 0), (0, -1), (1, 0), (0, 1)],  # Promoted Bishop (Player 1, Dragon Horse)
    [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)],  # King (Player 2)
    [(1, 0), (1, 1), (1, -1), (0, 1), (0, -1), (-1, 0)],  # Gold General (Player 2, regular)
    [(1, 0)],  # Pawn (Player 2)
    [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (0, -1)],  # Promoted Pawn (Player 2, Gold General)
    [(1, 1), (1, -1), (1, 0), (-1, 1), (-1, -1)],  # Silver General (Player 2)
    [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (0, -1)],  # Promoted Silver General (Player 2, Gold General)
    [(1, 0), (0, 1), (-1, 0), (0, -1)],  # Rook (Player 2)
    [(1, 1), (-1, 1), (-1, -1), (1, -1)],  # Promoted Rook (Player 2, Dragon King)
    [(1, 1), (-1, -1), (1, -1), (-1, 1)],  # Bishop (Player 2)
    [(1, 0), (0, 1), (-1, 0), (0, -1)],  # Promoted Bishop (Player 2, Dragon Horse)
]

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
    captured_pieces_1 = cython.declare(cython.list, visibility="public")
    captured_pieces_2 = cython.declare(cython.list, visibility="public")
    king_1 = cython.declare(cython.tuple[cython.int, cython.int], visibility="public")
    king_2 = cython.declare(cython.tuple[cython.int, cython.int], visibility="public")

    # Private fields
    check = cython.declare(cython.bint, visibility="public")
    # TODO DRAWS bepalen, na 4 keer zelfde positie, na herhaaldelijk check (anders dan schaak!)
    winner: cython.int

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
    ):
        """
        Initialize the game state with the given board configuration. If no board is provided, the game starts with the default setup.

        :param board: An optional board configuration.
        :param player: The player whose turn it is (1 or 2).
        """
        if board is None:
            self.board = self._init_board()
            self.board_hash = self._init_hash()
            # Keep track of the pieces on the board, this keeps evaluation and move generation from recomputing these over and over
            self.last_action = (-1, -1, -1, -1)
            self.player = 1

            self.captured_pieces_1 = []
            self.captured_pieces_2 = []
            self.current_player_moves = []
        else:
            self.board = board
            self.player = player
            self.board_hash = board_hash
            self.last_action = last_action
            self.captured_pieces_1 = captured_pieces_1
            self.captured_pieces_2 = captured_pieces_2
            self.king_1 = king_1
            self.king_2 = king_2

        # This needs to be determined after each move made.
        self.check = False
        self.winner = -2  # -2 means that the winner was not checked yet

    @cython.ccall
    def _init_board(self):
        board = np.zeros((5, 5), dtype=np.int32)

        # Set up the pieces for White (Player 2, top of the board)
        board[0] = [
            PIECES["K"] + P2_OFFS,  # King
            PIECES["G"] + P2_OFFS,  # Gold General
            PIECES["S"] + P2_OFFS,  # Silver General
            PIECES["B"] + P2_OFFS,  # Bishop
            PIECES["R"] + P2_OFFS,  # Rook
        ]
        board[1, 0] = PIECES["P"] + P2_OFFS  # Pawn in front of the king

        # Set up the pieces for Black (Player 1, bottom of the board)
        board[4] = [
            PIECES["R"],  # Rook
            PIECES["B"],  # Bishop
            PIECES["S"],  # Silver General
            PIECES["G"],  # Gold General
            PIECES["K"],  # King
        ]
        board[3, 4] = PIECES["P"]  # Pawn in front of the king

        self.king_1 = (4, 4)  # Black's king
        self.king_2 = (0, 0)  # White's king

        return board

    @cython.ccall
    def _init_hash(self):
        #  When a promoted piece is captured, it loses its promoted status.
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
            player=3 - self.player,
            board_hash=self.board_hash,
            last_action=self.last_action,
        )

    @cython.cfunc
    def apply_action_playout(self, action: cython.tuple) -> cython.void:
        assert self.winner <= -1, "Cannot apply action to a terminal state."

        from_row, from_col, to_row, to_col = action

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
            # Handle promotion
            assert self.is_promotion(from_row, from_col), "Cannot promote this piece."

            promoted_piece = self.get_promoted_piece(from_row, from_col)
            self.board[from_row, from_col] = promoted_piece

        else:
            # Handle move action
            piece = self.board[from_row, from_col]
            captured_piece = self.board[to_row, to_col]

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
            if piece == PIECES["K"]:  # Check if player 1's king is moved
                self.king_1 = (to_row, to_col)
            elif piece == PIECES["K"] + P2_OFFS:  # Check if player 2's king is moved
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
        self.winner = -2  # Reset the winner

    @cython.ccall
    def apply_action(self, action: cython.tuple) -> MiniShogi:
        assert self.winner <= -1, "Cannot apply action to a terminal state."

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
        )

        from_row, from_col, to_row, to_col = action

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
            # Handle promotion
            assert new_state.is_promotion(from_row, from_col), "Cannot promote this piece."

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
        if new_state.check:
            print(colored(f"Player {new_state.player} is in check", "red"))

        new_state.winner = -2  # Reset the winner
        return new_state

    @cython.cfunc
    def get_random_action(self) -> cython.tuple:
        assert self.winner <= -1, "Cannot get legal actions for a terminal state."

        drops = []
        moves = []
        captures = []
        promotions = []
        move_found = False
        player_piece_start, player_piece_end = (1, 10) if self.player == 1 else (11, 20)

        # Define callback for move generation
        def move_callback(from_row, from_col, to_row, to_col, piece):
            nonlocal move_found
            if self.board[to_row, to_col] != 0:
                captures.append((from_row, from_col, to_row, to_col))
            else:
                moves.append((from_row, from_col, to_row, to_col))

            move_found = True
            return False  # Continue generating moves

        # Handle move and promotion actions
        for row in range(5):
            for col in range(5):
                piece = self.board[row, col]
                # Piece belongs to the current player
                if player_piece_start <= piece <= player_piece_end:
                    # Check if the move is a promotion, promotion can never break check
                    if not self.check and self.is_promotion(row, piece):
                        promotions.append((row, col, row, col))
                        move_found = True

                    # Pawn on the last rows cannot move
                    if not (piece == PIECES["P"] and row == 4) or (piece == PIECES["P"] + P2_OFFS and row == 0):
                        # Use _generate_moves with random flag
                        self._generate_moves(row, col, piece, move_callback, randomize=True)

        # Handle drop actions
        captured_pieces = self.captured_pieces_1 if self.player == 1 else self.captured_pieces_2
        for piece in captured_pieces:
            for row in range(5):
                for col in range(5):
                    if self.board[row, col] == 0:  # Empty square
                        if self.is_legal_drop(row, col, piece, self.player):
                            if self.check:
                                if not self.simulate_move_exposes_king(-1, piece, row, col):
                                    drops.append((-1, piece, row, col))
                                    move_found = True
                            else:
                                drops.append((-1, piece, row, col))
                                move_found = True

        # If there are no legal actions, then we lost
        if not move_found:
            self.winner = 3 - self.player
        else:
            # -1 means that the winner is checked, but no winner has been determined yet
            self.winner = -1

        # Choose the action to return, prioritize captures, drops, promotions, and then moves
        if captures:
            return captures[randint(0, len(captures) - 1)]
        if drops:
            return drops[randint(0, len(drops) - 1)]
        if promotions:
            return promotions[randint(0, len(promotions) - 1)]
        if moves:
            return moves[randint(0, len(moves) - 1)]

        return None  # No legal actions available

    @cython.ccall
    def get_legal_actions(self) -> cython.list:
        assert self.winner <= -1, "Cannot get legal actions for a terminal state."

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
                    if not (piece == PIECES["P"] and row == 4) or (piece == PIECES["P"] + P2_OFFS and row == 0):
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

        # If there are no legal actions, then we lost
        if len(legal_actions) == 0:
            self.winner = 3 - self.player
        else:
            # -1 means that the winner is checked, but no winner has been determined yet
            self.winner = -1

        return legal_actions

    @cython.ccall
    def simulate_move_exposes_king(self, from_row, from_col, to_row, to_col) -> cython.bint:
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
        print(
            f"Checking if king for player: {player} will be attacked after move: {from_row, from_col, to_row, to_col}"
        )
        exposes_king = self.is_king_attacked(player)

        # Restore the original state
        self.board[to_row, to_col] = original_piece

        if not is_drop:
            self.board[from_row, from_col] = moving_piece

        return exposes_king

    @cython.ccall
    def is_legal_drop(self, row: cython.int, col: cython.int, piece: cython.int, player: cython.int) -> cython.bint:
        # Check if the target square is empty
        if self.board[row, col] != 0:
            return False  # We can only drop on unoccupied squares

        # Pawn-specific rules
        if piece == PIECES["P"] or piece == PIECES["P"] + P2_OFFS:
            # Check for another unpromoted pawn in the same column and last row restriction
            for r in range(5):
                if self.board[r, col] == piece or (player == 1 and row == 4) or (player == 2 and row == 0):
                    return False
            # Temporarily drop the piece
            self.board[row, col] = piece
            # Check if this results in checkmate
            checkmate = self.is_checkmate(3 - player)
            # Undo the drop
            self.board[row, col] = 0
            # Dropping a pawn may not result in checkmate
            return not checkmate
        # In any other case, a piece can move
        return True

    @cython.ccall
    def is_checkmate(self, player: cython.int) -> cython.bint:
        """
        Check if the given player is in checkmate.

        Args:
        player (cython.int): The player to check for checkmate.

        Returns:
        cython.bint: 1 if the player is in checkmate, otherwise 0.
        """
        if not self.is_king_attacked(player):
            return 0
        # Check for any legal move by the pieces
        for row in range(5):
            for col in range(5):
                piece = self.board[row, col]
                if (player == 1 and 1 <= piece <= 10) or (player == 2 and 11 <= piece <= 20):
                    if self.has_legal_move(row, col, piece, player):
                        return 0  # Found a legal move

        # Check for any legal drop that can resolve the check
        captured_pieces = self.captured_pieces_1 if player == 1 else self.captured_pieces_2
        for piece in captured_pieces:
            for row in range(5):
                for col in range(5):
                    if self.is_legal_drop(row, col, piece, player):
                        # We have to determine whether this move relieves the check
                        if not self.simulate_move_exposes_king(-1, piece, row, col):
                            return 0
        return 1  # No legal moves or drops, so it's checkmate

    @cython.ccall
    def _generate_moves(
        self, row: cython.int, col: cython.int, piece: cython.int, callback, randomize=False, count_defense=False
    ):
        player_piece_start, player_piece_end = (1, 10) if piece <= P2_OFFS else (11, 20)

        if 7 <= piece <= 10 or 17 <= piece <= 20:
            base_piece = PIECES["R"] if piece == 7 or piece == 8 or piece == 17 or piece == 18 else PIECES["B"]
            if piece >= 11:
                base_piece += P2_OFFS

            move_list = MOVES[base_piece]
            start_index = randint(0, len(move_list) - 1) if randomize else 0
            for i in range(len(move_list)):
                move = move_list[(i + start_index) % len(move_list)]
                new_row, new_col = row, col
                while True:
                    new_row += move[0]
                    new_col += move[1]
                    if not (0 <= new_row < 5 and 0 <= new_col < 5) or (
                        self.board[new_row, new_col] != 0
                        and player_piece_start <= self.board[new_row, new_col] <= player_piece_end
                    ):
                        if count_defense and player_piece_start <= self.board[new_row, new_col] <= player_piece_end:
                            # If defensive move, call the callback with an additional flag
                            if callback(row, col, new_row, new_col, piece, True):
                                return True

                        break  # Out of bounds or own piece encountered

                    if callback(row, col, new_row, new_col, piece):
                        return True  # Callback indicates to stop extending

                    if self.board[new_row, new_col] != 0:
                        break  # Opposing piece encountered, stop extending

        # Logic for standard and limited one-square moves
        move_list = MOVES[piece]
        start_index = randint(0, len(move_list) - 1) if randomize else 0
        for i in range(len(move_list)):
            move = move_list[(i + start_index) % len(move_list)]
            new_row, new_col = row + move[0], col + move[1]

            if not (0 <= new_row < 5 and 0 <= new_col < 5) or (
                self.board[new_row, new_col] != 0
                and player_piece_start <= self.board[new_row, new_col] <= player_piece_end
            ):
                if count_defense and player_piece_start <= self.board[new_row, new_col] <= player_piece_end:
                    # If defensive move, call the callback with an additional flag
                    if callback(row, col, new_row, new_col, piece, True):
                        return True
                continue  # Out of bounds or own piece encountered

            if callback(row, col, new_row, new_col, piece):
                return True  # Callback indicates to stop extending

        return False  # No valid move processed or no need to stop extending

    # TODO Change back to cfunc after debugging
    # @cython.cfunc
    def get_moves(self, row: cython.int, col: cython.int, moves: cython.list) -> cython.list:
        # ? Moeten de inner functies ook @cython.cfunc zijn?
        def move_callback(from_row, from_col, to_row, to_col, piece):
            if not self.simulate_move_exposes_king(from_row, from_col, to_row, to_col):
                moves.append((from_row, from_col, to_row, to_col))
            return False  # Continue generating all moves

        self._generate_moves(row, col, self.board[row, col], move_callback)
        return moves

    @cython.cfunc
    def is_king_attacked(self, player: cython.int) -> cython.bint:
        king_pos = self.king_1 if player == 1 else self.king_2

        def move_callback(from_row, from_col, to_row, to_col, piece):
            if (to_row, to_col) == king_pos:
                print(f"Move {from_row}, {from_col} attacks the king at {king_pos} of player {player} by {piece}")
            return (to_row, to_col) == king_pos  # Return True if king is attacked, this stops _generate_moves

        opposing_player_piece_start, opposing_player_piece_end = (11, 20) if player == 1 else (1, 10)
        for row in range(5):
            for col in range(5):
                piece = self.board[row, col]
                if opposing_player_piece_start <= piece <= opposing_player_piece_end:
                    if self._generate_moves(row, col, piece, move_callback):
                        print(f"{piece} at {row},{col} attacks king")
                        return True  # King is attacked
        return False

    @cython.cfunc
    def has_legal_move(self, row: cython.int, col: cython.int, piece: cython.int, player: cython.int) -> cython.bint:
        def move_callback(from_row, from_col, to_row, to_col, piece):
            # Return True if move is legal, this stops _generate_moves
            return not self.simulate_move_exposes_king(from_row, from_col, to_row, to_col)

        return self._generate_moves(row, col, piece, move_callback)

    @cython.ccall
    def is_terminal(self) -> cython.bint:
        """
        Check if the current game state is terminal (i.e., a player has won).

        :return: True if the game state is terminal, False otherwise.
        """
        # Any number higher than -1 means that we have found a winner.
        if self.winner > -1:
            return 1
        # -2 means that we did not yet check whether we are in a terminal state, so if we are in check, the state may be terminal
        if self.winner == -2 and self.check:
            return len(self.get_legal_actions()) == 0

        return 0

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

    default_params = array.array("d", [20.0, 1.0, 1.0, 5.0, 1.0, 80.0])

    @cython.cfunc
    @cython.exceptval(-9999999, check=False)
    def evaluate(
        self,
        player: cython.int,
        params: cython.double[:],
        norm: cython.bint = 0,
    ) -> cython.double:
        attacks = 0
        captures = 0
        defenses = 0
        board_control = [0] * 25
        material_score = 0
        score = 0

        def callback(from_row, from_col, to_row, to_col, piece, is_defense=False):
            # * Note that piece here, is the piece on the from_row/col so the piece to move
            nonlocal attacks, defenses, board_control, material_score
            multip: cython.int = (
                1 if player == 1 and 1 <= piece <= P2_OFFS or player == 2 and 11 <= piece <= 20 else -1
            )

            if self.board[to_row, to_col] != 0:
                if not is_defense:
                    attacks += multip * MATERIAL[self.board[to_row, to_col]]  # I can capture a piece
                else:
                    # Because the king has a score of 0 we exclude the king here
                    defenses += multip * MATERIAL[self.board[to_row, to_col]]  # Defending my piece

            # Keep track of the control of different squares
            board_control[to_row * 5 + to_col] += multip

            # Material count
            material_score += multip * MATERIAL[piece]

            return False  # Continue generating moves

        # Iterate through the board and use _generate_moves to count attacks, control and defenses
        for row in range(5):
            for col in range(5):
                if self.board[row, col] != 0:
                    self._generate_moves(row, col, self.board[row, col], callback, count_defense=True)

        # Count captured pieces
        for piece in self.captured_pieces_1 if player == 1 else self.captured_pieces_2:
            captures += MATERIAL[piece]
        for piece in self.captured_pieces_2 if player == 1 else self.captured_pieces_1:
            captures -= MATERIAL[piece]

        # Parameters:
        #  "m_check": 0,
        #  "m_attacks": 1,
        #  "m_captures": 2,
        #  "m_material": 3,
        #  "m_dominance": 4,
        #  "m_defenses": 5,
        #  "a": 6,

        # If the player to move is the player being evaluated and is in check, subtract the check penalty
        if self.check and player == self.player:
            score = -params[0]
        elif self.check and player != self.player:
            score = params[0]  # It's check for the opponent so it's good

        # Calculate the final score using parameters
        score += (
            attacks * params[1]
            + captures * params[2]
            + material_score * params[3]
            + sum(board_control) * params[4]
            + defenses * params[5]
        )

        if norm:
            return normalize(score, params[6])
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

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def evaluate_move(self, move: cython.tuple) -> cython.int:
        """
        Evaluate the given move using a simple heuristic for move ordering

        :param move: The move to evaluate.
        :return: The heuristic value of the move.
        """
        from_row, from_col, to_row, to_col = move
        moving_piece = self.board[from_row, from_col]
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
        for direction in MOVES[moving_piece]:
            def_row, def_col = to_row + direction[0], to_col + direction[1]
            if 0 <= def_row < 5 and 0 <= def_col < 5:
                if self.board[def_row, def_col] != 0:
                    if self.is_same_player_piece(moving_piece, self.board[def_row, def_col]):
                        score += MATERIAL[self.board[def_row, def_col]] / 4.0  # Defensive move
                    else:
                        score += MATERIAL[self.board[def_row, def_col]] / 2.0  # Offensive move

        return cython.cast(cython.int, score)

    @cython.cfunc
    @cython.inline
    def is_same_player_piece(self, piece1, piece2) -> cython.bint:
        """
        Check if two pieces belong to the same player.

        :param piece1: First piece.
        :param piece2: Second piece.
        :return: True if both pieces belong to the same player, otherwise False.
        """
        return (piece1 <= P2_OFFS and piece2 <= P2_OFFS) or (piece1 > P2_OFFS and piece2 > P2_OFFS)

    @cython.ccall
    def visualize(self, full_debug=False) -> str:
        """
        Visualize the Minishogi board with additional debug information.

        :param full_debug: Whether to show full debug information (like board hash).
        :return: String representation of the board and additional information.
        """

        board_str = "Minishogi Board:\n"
        for i in range(5):
            for j in range(5):
                piece = self.board[i, j]

                char_index = piece % P2_OFFS - 1  # Adjust according to your piece encoding
                is_promoted = (piece > 3 and piece % 2 == 0) or (piece > 13 and piece % 2 == 0)
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
            board_str += f"Check: {'Yes' if self.check else 'No'}\n"
            board_str += f"Captured Pieces Player 1: {self.captured_pieces_1}\n"
            board_str += f"Captured Pieces Player 2: {self.captured_pieces_2}\n"

            # Evaluations for each player
            p1_eval = self.evaluate(1, self.default_params)
            p2_eval = self.evaluate(2, self.default_params)
            board_str += f"Player 1 Evaluation: {p1_eval}\n"
            board_str += f"Player 2 Evaluation: {p2_eval}\n"

            # All legal actions
            legal_actions = self.get_legal_actions()
            board_str += f"Legal Actions: {legal_actions}\n"

            # Move evaluations (assuming you have a function for this)
            move_evaluations = [self.evaluate_move(move) for move in legal_actions]
            board_str += f"Move Evaluations: {move_evaluations}\n"

        return board_str

    @property
    def transposition_table_size(self):
        # return an appropriate size based on the game characteristics
        return 2**20

    def __repr__(self) -> str:
        game: str = "minishogi"
        return game

    @cython.cfunc
    @cython.inline
    def is_promotion(self, row: cython.int, piece: cython.int) -> cython.bint:
        # Check if piece is unpromoted and promotable
        if (piece > 2 and piece % 2 != 0) or (piece > 12 and piece % 2 != 0):
            if piece < 11:  # Player 1's pieces
                return row == 4
            else:  # Player 2's pieces
                return row == 0
        return False

    @cython.cfunc
    @cython.inline
    @cython.exceptval(-1)
    def get_promoted_piece(self, row: cython.int, col: cython.int) -> cython.int:
        piece: cython.int = self.board[row, col]

        # Promote if piece is unpromoted
        if (piece > 2 and piece % 2 != 0) or (piece > 12 and piece % 2 != 0):
            return piece + 1

        return piece

    @cython.cfunc
    @cython.inline
    @cython.exceptval(-1)
    def get_demoted_piece(self, piece: cython.int) -> cython.int:
        # Demote if piece is promoted
        if (piece > 3 and piece % 2 == 0) or (piece > 13 and piece % 2 == 0):
            return piece - 1

        return piece

    # @cython.ccall
    # @cython.locals(player=cython.int)
    # def update_threatened_pieces(self) -> None:
    #     # TODO Volgens mij is dit toch meer iets voor de evaluatie, omdat je de checkmate ook al in de get_legal_actions kan doen.
    #     # Clear the current list of threatened pieces for both players
    #     self.threatened_pieces_1.clear()
    #     self.threatened_pieces_2.clear()

    #     # Reset the check flag
    #     self.check = False

    #     # Iterate over both players
    #     for player in range(1, 3):
    #         # Identify the opposing player
    #         opposing_player: cython.int = 2 if player == 1 else 1

    #         # Get all legal actions for the opposing player
    #         opposing_moves: cython.list = self.get_legal_actions(self.board, opposing_player)

    #         # While we have generated the moves, we might as well store them.
    #         if opposing_player == self.player:
    #             self.current_player_moves = opposing_moves

    #         for move in opposing_moves:
    #             # Extract the destination position from the move
    #             dest_row: cython.int = move[2]
    #             dest_col: cython.int = move[3]

    #             # Get the piece at the destination
    #             target_piece: cython.int = self.board[dest_row][dest_col]

    #             # Check if the destination contains a piece of the current player
    #             if (player == 1 and 11 <= target_piece <= 20) or (player == 2 and 1 <= target_piece <= P2_OFFS):
    #                 # Add the threatened piece to the list
    #                 if player == 1:
    #                     self.threatened_pieces_1.append((dest_row, dest_col, target_piece))
    #                 else:
    #                     self.threatened_pieces_2.append((dest_row, dest_col, target_piece))

    #                 # Check if the threatened piece is a king
    #                 if target_piece == PIECES["K"] or target_piece == PIECES["K"] + P2_OFFS:
    #                     # Set the check flag only if the current player's king is threatened
    #                     if self.player == player:
    #                         self.check = True
    #                     else:
    #                         assert False, "Opponent's king is threatened, this should not be possible."

    # @cython.ccall
    # def get_moves(self, row: cython.int, col: cython.int, moves: cython.list) -> cython.list:
    #     piece = self.board[row][col]
    #     player_piece_start, player_piece_end = (1, 10) if piece <= 10 else (11, 20)

    #     # Generate unlimited moves for Rook, Bishop, Dragon King, and Dragon Horse
    #     if 7 <= piece <= P2_OFFS or 17 <= piece <= 20:
    #         base_piece = PIECES["R"] if piece == 7 or piece == 8 or piece == 17 or piece == 18 else PIECES["B"]

    #         # This takes care of the direction of the moves
    #         if piece >= P2_OFFS:
    #             base_piece += P2_OFFS

    #         # Add the moves for the base piece
    #         for move in MOVES[base_piece]:
    #             new_row, new_col = row, col
    #             while True:
    #                 new_row += move[0]
    #                 new_col += move[1]
    #                 if not (0 <= new_row < 5 and 0 <= new_col < 5):
    #                     break
    #                 if self.board[new_row][new_col] != 0:
    #                     if player_piece_start <= self.board[new_row][new_col] <= player_piece_end:
    #                         break  # We've encountered one of our own pieces
    #                     elif not self.simulate_move_exposes_king(row, col, new_row, new_col):
    #                         # The move does not expose the king to an attack
    #                         moves.append((row, col, new_row, new_col))
    #                     break  # We've encountered an opposing piece, so we can't move further
    #                 elif not self.simulate_move_exposes_king(row, col, new_row, new_col):
    #                     # The move does not expose the king to an attack
    #                     moves.append((row, col, new_row, new_col))

    #     # Generate the standard and limited one-square moves for all pieces
    #     # For dragon horse and dragon king, the MOVES dictionary contains ONLY the extra moves
    #     for move in MOVES[piece]:
    #         new_row, new_col = row + move[0], col + move[1]
    #         if 0 <= new_row < 5 and 0 <= new_col < 5:
    #             if self.board[new_row][new_col] == 0 or not (
    #                 player_piece_start <= self.board[new_row][new_col] <= player_piece_end
    #             ):
    #                 if not self.simulate_move_exposes_king(row, col, new_row, new_col):
    #                     moves.append((row, col, new_row, new_col))

    #     return moves

    # @cython.ccall
    # def is_king_attacked(self, player: cython.int) -> bool:
    #     king_pos = self.king_1 if player == 1 else self.king_2
    #     opposing_player_piece_start, opposing_player_piece_end = (11, 20) if player == 1 else (1, 10)

    #     for row in range(5):
    #         for col in range(5):
    #             piece = self.board[row][col]
    #             if opposing_player_piece_start <= piece <= opposing_player_piece_end:
    #                 # Generate moves for Rook, Bishop, Dragon King, Dragon Horse
    #                 if 7 <= piece <= P2_OFFS or 17 <= piece <= 20:
    #                     base_piece = PIECES["R"] if piece in [7, 8, 17, 18] else PIECES["B"]
    #                     if piece >= P2_OFFS:
    #                         base_piece += P2_OFFS

    #                     for move in MOVES[base_piece]:
    #                         new_row, new_col = row, col

    #                         while True:
    #                             new_row += move[0]
    #                             new_col += move[1]

    #                             if not (0 <= new_row < 5 and 0 <= new_col < 5):
    #                                 break
    #                             if (new_row, new_col) == king_pos:
    #                                 return True
    #                             if self.board[new_row][new_col] != 0:
    #                                 break

    #                 # Generate standard and limited one-square moves for all pieces
    #                 for move in MOVES[piece]:
    #                     new_row, new_col = row + move[0], col + move[1]
    #                     if 0 <= new_row < 5 and 0 <= new_col < 5:
    #                         if (new_row, new_col) == king_pos:
    #                             return True
    #                         if self.board[new_row][new_col] != 0:
    #                             break

    #     return False

    # @cython.ccall
    # def has_legal_move(self, row: cython.int, col: cython.int, piece: cython.int, player: cython.int) -> cython.bint:
    #     """
    #     Check if a piece has any legal moves.

    #     Args:
    #     row (cython.int): The row of the piece.
    #     col (cython.int): The column of the piece.
    #     piece (cython.int): The piece to check for legal moves.
    #     player (cython.int): The player who owns the piece.

    #     Returns:
    #     cython.bint: 1 if there's at least one legal move, otherwise 0.
    #     """
    #     player_piece_start, player_piece_end = (1, 10) if player == 1 else (11, 20)

    #     # Generate unlimited moves for Rook, Bishop, Dragon King, and Dragon Horse
    #     if 7 <= piece <= P2_OFFS or 17 <= piece <= 20:
    #         base_piece = PIECES["R"] if piece in [7, 8, 17, 18] else PIECES["B"]

    #         if piece >= P2_OFFS:
    #             base_piece += P2_OFFS

    #         for move in MOVES[base_piece]:
    #             new_row, new_col = row, col
    #             while True:
    #                 new_row += move[0]
    #                 new_col += move[1]
    #                 if not (0 <= new_row < 5 and 0 <= new_col < 5):
    #                     break
    #                 if self.board[new_row][new_col] != 0 and (
    #                     player_piece_start <= self.board[new_row][new_col] <= player_piece_end
    #                 ):
    #                     break  # Encountered own piece, can't move further
    #                 if not self.simulate_move_exposes_king(row, col, new_row, new_col):
    #                     return 1  # Found a legal move

    #     # Generate standard and limited one-square moves for all pieces
    #     for move in MOVES.get(piece, []):
    #         new_row, new_col = row + move[0], col + move[1]
    #         if 0 <= new_row < 5 and 0 <= new_col < 5:
    #             if self.board[new_row][new_col] != 0 and (
    #                 player_piece_start <= self.board[new_row][new_col] <= player_piece_end
    #             ):
    #                 continue  # Encountered own piece, skip this move
    #             if not self.simulate_move_exposes_king(row, col, new_row, new_col):
    #                 return 1  # Found a legal move

    #     return 0


def get_upside_down_char(char):
    # Mapping of regular characters to their upside-down Unicode equivalents
    upside_down_map = {
        "K": "ʞ",
        "p": "q",
        "P": "d",
        "s": "s",
        "S": "S",
        "r": "ɹ",
        "R": "ᴚ",
        "b": "q",
        "B": "d",
        "G": "⅁",
    }
    return upside_down_map.get(char, char)  # Return the upside-down char if exists, else return the char itself

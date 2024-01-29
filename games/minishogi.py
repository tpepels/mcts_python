# cython: language_level=3, wraparound=False

import cython

from fastrand import pcg32randint as randint
import numpy as np

from cython.cimports.includes import normalize, GameState, win, loss, draw, f_index
from cython.cimports.games.minishogi import MOVES, PIECES, PIECE_CHARS

MOVES = {
    1: [  # King (Player 1)
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ],
    2: [  # Gold General (Player 1, regular)
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
    ],
    3: [  # Pawn (Player 1)
        (1, 0),
    ],
    4: [  # Promoted Pawn (Player 1, Gold General)
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
    ],
    5: [  # Silver General (Player 1)
        (1, 1),
        (1, -1),
        (0, 1),
        (-1, 1),
        (-1, -1),
    ],
    6: [  # Promoted Silver General (Player 1, Gold General)
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
    ],
    7: [  # Rook (Player 1)
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
    ],
    8: [  # Promoted Rook (Player 1, Dragon King)
        (1, 1),
        (-1, 1),
        (-1, -1),
        (1, -1),
    ],
    9: [  # Bishop (Player 1)
        (1, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
    ],
    10: [  # Promoted Bishop (Player 1, Dragon Horse)
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
    ],
    11: [  # King (Player 2)
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
    ],
    12: [  # Gold General (Player 2, regular)
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (0, 1),
    ],
    13: [  # Pawn (Player 2)
        (-1, 0),
    ],
    14: [  # Promoted Pawn (Player 2, Gold General)
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (0, 1),
    ],
    15: [  # Silver General (Player 2)
        (-1, -1),
        (-1, 1),
        (0, -1),
        (1, -1),
        (1, 1),
    ],
    16: [  # Promoted Silver General (Player 2, Gold General)
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (0, 1),
    ],
    17: [  # Rook (Player 2)
        (-1, 0),
        (0, -1),
        (1, 0),
        (0, 1),
    ],
    18: [  # Promoted Rook (Player 2, Dragon King)
        (-1, -1),
        (1, -1),
        (1, 1),
        (-1, 1),
    ],
    19: [  # Bishop (Player 2)
        (-1, -1),
        (1, 1),
        (-1, 1),
        (1, -1),
    ],
    20: [  # Promoted Bishop (Player 2, Dragon Horse)
        (-1, 0),
        (0, -1),
        (1, 0),
        (0, 1),
    ],
}


P2_OFFS = 10

PIECES = {
    "king": 1,
    "gold general": 2,  # Regular gold general (cannot be promoted)
    "pawns": 3,
    "+p": 4,  # An upgraded pawn is a gold general
    "silver general": 5,
    "+s": 6,  # A promoted silver general is a gold general
    "rook": 7,
    "+r": 8,  # A promoted rook is a dragon king
    "bishop": 9,
    "+b": 10,  # A promoted bishop is a dragon horse
}

# Define the pieces for both players, including promotions
PIECE_CHARS = ["K", "G", "p", "P", "s", "S", "r", "R", "b", "B"]


class MiniShogi(GameState):
    zobrist_table = [[[randint(1, 2**61 - 1) for _ in range(21)] for _ in range(5)] for _ in range(5)]
    REUSE = True
    board = cython.declare(cython.int[:], visibility="public")
    board_hash = cython.declare(cython.longlong, visibility="public")
    last_action = cython.declare(cython.tuple[cython.int, cython.int], visibility="public")
    current_player_moves = cython.declare(cython.list, visibility="public")
    captured_pieces_1 = cython.declare(cython.list, visibility="public")
    captured_pieces_2 = cython.declare(cython.list, visibility="public")
    threatened_pieces_1 = cython.declare(cython.list, visibility="public")
    threatened_pieces_2 = cython.declare(cython.list, visibility="public")
    king_1 = cython.declare(cython.tuple[cython.int, cython.int], visibility="public")
    king_2 = cython.declare(cython.tuple[cython.int, cython.int], visibility="public")
    # Private fields
    check: cython.bint
    # TODO DRAWS bepalen
    winner: cython.int

    def __init__(
        self,
        board=None,
        player=1,
        board_hash=0,
        last_action=(0, 0),
    ):
        """
        Initialize the game state with the given board configuration.
        If no board is provided, the game starts with the default setup.

        :param board: An optional board configuration.
        :param player: The player whose turn it is (1 or 2).
        """
        if board is None:
            self.board = self._init_board()
            self.board_hash = self._init_hash()
            # Keep track of the pieces on the board, this keeps evaluation and move generation from recomputing these over and over
            self.last_action = (-1, -1)
            self.player = 1
            self.winner = -1

            self.captured_pieces_1 = []
            self.captured_pieces_2 = []
            self.threatened_pieces_1 = []
            self.threatened_pieces_2 = []
            self.current_player_moves = []
        else:
            self.board = board
            self.player = player
            self.board_hash = board_hash
            self.last_action = last_action

        # This needs to be determined after each move made.
        self.check = False

    def _init_board(self):
        board = np.zeros((5, 5), dtype=np.int32)
        # Set up the pieces for Player 1 (bottom of the board)
        board[4] = [
            PIECES["king"],
            PIECES["gold general"],
            PIECES["silver general"],
            PIECES["bishop"],
            PIECES["rook"],
        ]
        board[3][0] = PIECES["pawns"]  # Pawn in front of the king
        # Set up the pieces for Player 2 (top of the board)
        # Player 2's pieces are offset by 9 (1-9 for Player 1, 11-19 for Player 2)

        board[0] = [
            PIECES["king"] + P2_OFFS,
            PIECES["gold general"] + P2_OFFS,
            PIECES["silver general"] + P2_OFFS,
            PIECES["bishop"] + P2_OFFS,
            PIECES["rook"] + P2_OFFS,
        ]
        board[1][0] = PIECES["pawns"] + P2_OFFS  # Pawn in front of the king

        self.king_1 = (4, 0)
        self.king_2 = (0, 0)

        return board

    def _init_hash(self):
        #  When a promoted piece is captured, it loses its promoted status.
        # If that piece is later dropped back onto the board by the player who captured it,
        # it returns as an unpromoted piece.
        # This means that we do not have to keep track of the promoted status of captured pieces.
        hash_value = 0
        for row in range(5):
            for col in range(5):
                piece = self.board[row][col]
                # Use the piece value directly as the index in the Zobrist table
                hash_value ^= MiniShogi.zobrist_table[row][col][piece]
        return hash_value

    @cython.ccall
    def apply_action(self, action: cython.tuple) -> MiniShogi:
        assert self.winner == -1, "Cannot apply action to a terminal state."

        # Create a copy of the current game state
        new_state = MiniShogi(
            board=self.board.copy(),
            player=self.player,
            board_hash=self.board_hash,
            last_action=self.last_action,
        )

        from_row, from_col, to_row, to_col = action

        if from_row == -1:
            # Handle drop action
            # ! This assumes that pieces are demoted on capture, i.e. the capture lists have only unpromoted pieces
            piece = from_col
            new_state.board[to_row][to_col] = piece
            new_state.board_hash ^= MiniShogi.zobrist_table[to_row][to_col][piece]

            # Remove the piece from the captured pieces list
            if new_state.player == 1:
                new_state.captured_pieces_1.remove(piece)
            else:
                new_state.captured_pieces_2.remove(piece)
        elif from_row == to_row and from_col == to_col:
            # Handle promotion
            assert self.is_promotion(from_row, from_col), "Cannot promote this piece."

            piece = new_state.board[from_row][from_col]
            promoted_piece = self.get_promoted_piece(from_row, from_col)

            new_state.board_hash ^= MiniShogi.zobrist_table[from_row][from_col][piece]
            new_state.board_hash ^= MiniShogi.zobrist_table[from_row][from_col][promoted_piece]
            new_state.board[from_row][from_col] = promoted_piece

        else:
            # Handle move action
            piece = new_state.board[from_row][from_col]
            captured_piece = new_state.board[to_row][to_col]

            new_state.board_hash ^= MiniShogi.zobrist_table[from_row][from_col][piece]
            new_state.board_hash ^= MiniShogi.zobrist_table[to_row][to_col][piece]

            if captured_piece != 0:
                new_state.board_hash ^= MiniShogi.zobrist_table[to_row][to_col][captured_piece]
                # Demote the piece if it is captured (after updating the hash)
                captured_piece = self.get_demoted_piece(captured_piece)
                # Put the piece in the opposing player's captured pieces list, so they can drop them later
                if new_state.player == 1:
                    # Flip the piece to the opposing player's perspective
                    new_state.captured_pieces_2.append(captured_piece - P2_OFFS)
                else:
                    # Flip the piece to the opposing player's perspective
                    new_state.captured_pieces_1.append(captured_piece + P2_OFFS)

            # Update the king's position if a king is moved
            if piece == PIECES["king"]:  # Check if player 1's king is moved
                new_state.king_1 = (to_row, to_col)
            elif piece == PIECES["king"] + P2_OFFS:  # Check if player 2's king is moved
                new_state.king_2 = (to_row, to_col)

            # This will be the promoted piece if the move is a promotion, otherwise it will be the original piece
            new_state.board[to_row][to_col] = piece
            new_state.board[from_row][from_col] = 0

        # Update the last action
        new_state.last_action = action

        # Switch player after the action
        new_state.player = 1 if new_state.player == 2 else 2

        # After each move, determine if I am in check
        self.check = self.is_king_attacked(self.player)
        return new_state

    @cython.ccall
    def get_legal_actions(self) -> cython.list:
        assert self.winner == -1, "Cannot get legal actions for a terminal state."

        legal_actions = []
        player_piece_start, player_piece_end = (1, 9) if self.player == 1 else (10, 19)

        # Handle move and promotion actions
        for row in range(5):
            for col in range(5):
                piece = self.board[row][col]
                # Piece belongs to the current player
                if player_piece_start <= piece <= player_piece_end:
                    # Check if the move is a promotion, promotion can never break check
                    if not self.check and self.is_promotion(row, piece):
                        legal_actions.append((row, col, row, col))

                    # Pawns on the last rows cannot move (this makes sure not to generate moves that are not needed)
                    if not (piece == PIECES["pawns"] and row == 4) or (
                        piece == PIECES["pawns"] + P2_OFFS and row == 0
                    ):
                        # Generate moves for this piece
                        self.get_moves(row, col, piece, legal_actions)

        # Handle drop actions
        captured_pieces = self.captured_pieces_1 if self.player == 1 else self.captured_pieces_2

        for piece in captured_pieces:
            for row in range(5):
                for col in range(5):
                    if self.board[row][col] == 0:  # Empty square
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
            self.winner = -1

        return legal_actions

    @cython.ccall
    def simulate_move_exposes_king(self, from_row, from_col, to_row, to_col) -> bool:
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
        original_piece = self.board[to_row][to_col]
        moving_piece = from_col if is_drop else self.board[from_row][from_col]

        # Perform the action (move or drop)
        self.board[to_row][to_col] = moving_piece
        if not is_drop:
            self.board[from_row][from_col] = 0

        # Check if the king is attacked after this action
        player = 1 if moving_piece <= 9 else 2
        exposes_king = self.is_king_attacked(player)

        # Restore the original state
        self.board[to_row][to_col] = original_piece

        if not is_drop:
            self.board[from_row][from_col] = moving_piece

        return exposes_king

    @cython.ccall
    def is_legal_drop(self, row: cython.int, col: cython.int, piece: cython.int, player: cython.int) -> cython.bint:
        # Check if the target square is empty
        if self.board[row][col] != 0:
            return False  # We can only drop on unoccupied squares

        # Pawn-specific rules
        if piece == PIECES["pawns"] or piece == PIECES["pawns"] + P2_OFFS:
            # Check for another unpromoted pawn in the same column and last row restriction
            for r in range(5):
                if self.board[r][col] == piece or (player == 1 and row == 4) or (player == 2 and row == 0):
                    return False
            # Temporarily drop the piece
            self.board[row][col] = piece
            # Check if this results in checkmate
            checkmate = self.is_checkmate(3 - player)
            # Undo the drop
            self.board[row][col] = 0
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
                piece = self.board[row][col]
                if (player == 1 and 1 <= piece <= 9) or (player == 2 and 10 <= piece <= 19):
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
    def _generate_moves(self, row: cython.int, col: cython.int, piece: cython.int, callback):
        player_piece_start, player_piece_end = (1, 9) if piece <= 9 else (10, 19)

        # Logic for unlimited moves (e.g., Rook, Bishop, Dragon King, Dragon Horse)
        if piece in [7, 8, 17, 18]:
            base_piece = PIECES["rook"] if piece in [7, 8, 17, 18] else PIECES["bishop"]
            if piece >= 10:
                base_piece += P2_OFFS

            for move in MOVES[base_piece]:
                new_row, new_col = row, col
                while True:
                    new_row += move[0]
                    new_col += move[1]
                    if (
                        not (0 <= new_row < 5 and 0 <= new_col < 5)
                        or self.board[new_row][new_col] != 0
                        and player_piece_start <= self.board[new_row][new_col] <= player_piece_end
                    ):
                        break  # Out of bounds or own piece encountered

                    if callback(row, col, new_row, new_col, piece):
                        return True  # Callback indicates to stop extending

                    if self.board[new_row][new_col] != 0:
                        break  # Opposing piece encountered, stop extending

        # Logic for standard and limited one-square moves
        for move in MOVES[piece]:
            new_row, new_col = row + move[0], col + move[1]
            if (
                not (0 <= new_row < 5 and 0 <= new_col < 5)
                or self.board[new_row][new_col] != 0
                and player_piece_start <= self.board[new_row][new_col] <= player_piece_end
            ):
                continue  # Out of bounds or own piece encountered

            if callback(row, col, new_row, new_col, piece):
                return True  # Callback indicates to stop extending

        return False  # No valid move processed or no need to stop extending

    @cython.ccall
    def get_moves(self, row: cython.int, col: cython.int, moves: cython.list) -> cython.list:
        def move_callback(from_row, from_col, to_row, to_col, piece):
            if not self.simulate_move_exposes_king(from_row, from_col, to_row, to_col):
                moves.append((from_row, from_col, to_row, to_col))
            return False  # Continue generating all moves, i.e never return True as this stops _generate_moves

        self._generate_moves(row, col, self.board[row][col], move_callback)
        return moves

    @cython.ccall
    def is_king_attacked(self, player: cython.int) -> bool:
        king_pos = self.king_1 if player == 1 else self.king_2

        def move_callback(from_row, from_col, to_row, to_col, piece):
            return (to_row, to_col) == king_pos  # Return True if king is attacked, this stops _generate_moves

        opposing_player_piece_start, opposing_player_piece_end = (10, 19) if player == 1 else (1, 9)
        for row in range(5):
            for col in range(5):
                piece = self.board[row][col]
                if opposing_player_piece_start <= piece <= opposing_player_piece_end:
                    if self._generate_moves(row, col, piece, move_callback):
                        return True  # King is attacked
        return False

    @cython.ccall
    def has_legal_move(self, row: cython.int, col: cython.int, piece: cython.int, player: cython.int) -> cython.bint:
        def move_callback(from_row, from_col, to_row, to_col, piece):
            # Return True if move is legal, this stops _generate_moves
            return not self.simulate_move_exposes_king(from_row, from_col, to_row, to_col)

        return self._generate_moves(row, col, piece, move_callback)

    @cython.ccall
    @cython.inline
    def is_promotion(self, row: cython.int, piece: cython.int) -> bool:
        # Check if piece is unpromoted and promotable
        if (piece > 2 and piece % 2 != 0) or (piece > 12 and piece % 2 != 0):
            if piece < 10:  # Player 1's pieces
                return row == 4
            else:  # Player 2's pieces
                return row == 0
        return False

    @cython.ccall
    @cython.inline
    @cython.exceptval(-1)
    def get_promoted_piece(self, row: cython.int, col: cython.int) -> cython.int:
        piece: cython.int = self.board[row][col]

        # Promote if piece is unpromoted
        if (piece > 2 and piece % 2 != 0) or (piece > 12 and piece % 2 != 0):
            return piece + 1

        return piece

    @cython.ccall
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
    #             if (player == 1 and 10 <= target_piece <= 19) or (player == 2 and 1 <= target_piece <= 9):
    #                 # Add the threatened piece to the list
    #                 if player == 1:
    #                     self.threatened_pieces_1.append((dest_row, dest_col, target_piece))
    #                 else:
    #                     self.threatened_pieces_2.append((dest_row, dest_col, target_piece))

    #                 # Check if the threatened piece is a king
    #                 if target_piece == PIECES["king"] or target_piece == PIECES["king"] + 9:
    #                     # Set the check flag only if the current player's king is threatened
    #                     if self.player == player:
    #                         self.check = True
    #                     else:
    #                         assert False, "Opponent's king is threatened, this should not be possible."

    # @cython.ccall
    # def get_moves(self, row: cython.int, col: cython.int, moves: cython.list) -> cython.list:
    #     piece = self.board[row][col]
    #     player_piece_start, player_piece_end = (1, 9) if piece <= 9 else (10, 19)

    #     # Generate unlimited moves for Rook, Bishop, Dragon King, and Dragon Horse
    #     if 7 <= piece <= 10 or 17 <= piece <= 20:
    #         base_piece = PIECES["rook"] if piece == 7 or piece == 8 or piece == 17 or piece == 18 else PIECES["bishop"]

    #         # This takes care of the direction of the moves
    #         if piece >= 10:
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
    #     opposing_player_piece_start, opposing_player_piece_end = (10, 19) if player == 1 else (1, 9)

    #     for row in range(5):
    #         for col in range(5):
    #             piece = self.board[row][col]
    #             if opposing_player_piece_start <= piece <= opposing_player_piece_end:
    #                 # Generate moves for Rook, Bishop, Dragon King, Dragon Horse
    #                 if 7 <= piece <= 10 or 17 <= piece <= 20:
    #                     base_piece = PIECES["rook"] if piece in [7, 8, 17, 18] else PIECES["bishop"]
    #                     if piece >= 10:
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
    #     player_piece_start, player_piece_end = (1, 9) if player == 1 else (10, 19)

    #     # Generate unlimited moves for Rook, Bishop, Dragon King, and Dragon Horse
    #     if 7 <= piece <= 10 or 17 <= piece <= 20:
    #         base_piece = PIECES["rook"] if piece in [7, 8, 17, 18] else PIECES["bishop"]

    #         if piece >= 10:
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

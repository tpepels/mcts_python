# cython: language_level=3, wraparound=False

import cython

from fastrand import pcg32randint as randint
import numpy as np

from cython.cimports.includes import normalize, GameState, win, loss, draw, f_index
from cython.cimports.games.minishogi import MOVES, PIECES, PIECE_CHARS

MOVES = {
    1: [
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    ],  # King (Player 1)
    4: [(0, 1), (1, 1), (1, 0), (0, -1), (-1, 0), (-1, 1)],  # Gold General (Player 1)
    5: [(1, 1), (1, -1), (0, 1), (-1, 1), (-1, -1)],  # Silver General (Player 1)
    3: [(1, 1), (-1, -1), (1, -1), (-1, 1)],  # Bishop (Player 1)
    2: [(0, 1), (1, 0), (0, -1), (-1, 0)],  # Rook (Player 1)
    6: [(0, 1)],  # Pawn (Player 1)
    7: [
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    ],  # Promoted Rook (Dragon King, Player 1)
    8: [
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    ],  # Promoted Bishop (Dragon Horse, Player 1)
    9: [(0, 1), (1, 1), (1, 0), (0, -1), (-1, 0), (-1, 1)],  # Promoted Pawn (Player 1)
    10: [
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    ],  # King (Player 2)
    13: [(0, 1), (1, 1), (1, 0), (0, -1), (-1, 0), (-1, 1)],  # Gold General (Player 2)
    14: [(1, 1), (1, -1), (0, 1), (-1, 1), (-1, -1)],  # Silver General (Player 2)
    12: [(1, 1), (-1, -1), (1, -1), (-1, 1)],  # Bishop (Player 2)
    11: [(0, 1), (1, 0), (0, -1), (-1, 0)],  # Rook (Player 2)
    15: [(0, 1)],  # Pawn (Player 2)
    16: [
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    ],  # Promoted Rook (Dragon King, Player 2)
    17: [
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    ],  # Promoted Bishop (Dragon Horse, Player 2)
    18: [(0, 1), (1, 1), (1, 0), (0, -1), (-1, 0), (-1, 1)],  # Promoted Pawn (Player 2)
}

PIECES = {
    "king": 1,
    "rook": 2,
    "bishop": 3,
    "pawns": 4,
    "gold general": 5,
    "silver general": 6,
    "+r": 7,
    "+b": 8,
    "+p": 9,
}

# Define the pieces for both players, including promotions
PIECE_CHARS = ["K", "r", "b", "G", "S", "p", "R", "B", "P"]


class MiniShogi(GameState):
    zobrist_table = [
        [[randint(1, 2**61 - 1) for _ in range(19)] for _ in range(5)]
        for _ in range(5)
    ]
    REUSE = True
    board = cython.declare(cython.int[:], visibility="public")
    board_hash = cython.declare(cython.longlong, visibility="public")
    last_action = cython.declare(
        cython.tuple[cython.int, cython.int], visibility="public"
    )
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
        # Player 2's pieces are offset by 9 (1-9 for Player 1, 10-18 for Player 2)
        player_2_offset = 9
        board[0] = [
            PIECES["king"] + player_2_offset,
            PIECES["gold general"] + player_2_offset,
            PIECES["silver general"] + player_2_offset,
            PIECES["bishop"] + player_2_offset,
            PIECES["rook"] + player_2_offset,
        ]
        board[1][0] = PIECES["pawns"] + player_2_offset  # Pawn in front of the king

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
            board=[row[:] for row in self.board],  # Deep copy of the board
            player=self.player,
            board_hash=self.board_hash,
            last_action=self.last_action,
        )

        from_row, from_col, to_row, to_col = action

        if from_row == -1:
            # Handle drop action
            # ! This assumes that pieces are demoted on capture
            piece = from_col
            new_state.board[to_row][to_col] = piece
            new_state.board_hash ^= MiniShogi.zobrist_table[from_row][from_col][piece]
            new_state.board_hash ^= MiniShogi.zobrist_table[to_row][to_col][piece]

            # Remove the piece from the captured pieces list
            if new_state.player == 1:
                new_state.captured_pieces_1.remove(piece)
            else:
                new_state.captured_pieces_2.remove(piece)
        else:
            # TODO Handle promotions
            # Handle move action
            piece = new_state.board[from_row][from_col]
            captured_piece = new_state.board[to_row][to_col]

            new_state.board_hash ^= MiniShogi.zobrist_table[from_row][from_col][piece]
            new_state.board_hash ^= MiniShogi.zobrist_table[to_row][to_col][piece]

            if captured_piece != 0:
                new_state.board_hash ^= MiniShogi.zobrist_table[to_row][to_col][
                    captured_piece
                ]
                # Demote the piece if it is captured (after updating the hash)
                captured_piece = self.get_demoted_piece(captured_piece)
                if new_state.player == 1:
                    new_state.captured_pieces_2.append(captured_piece)
                else:
                    new_state.captured_pieces_1.append(captured_piece)

            # Update the king's position if a king is moved
            if piece == PIECES["king"]:  # Check if player 1's king is moved
                new_state.king_1 = (to_row, to_col)
            elif piece == PIECES["king"] + 9:  # Check if player 2's king is moved
                new_state.king_2 = (to_row, to_col)

            new_state.board[to_row][to_col] = piece
            new_state.board[from_row][from_col] = 0

        # Update the last action
        new_state.last_action = action

        # Switch player after the action
        new_state.player = 1 if new_state.player == 2 else 2

        return new_state

    @cython.ccall
    @cython.locals(player=cython.int)
    def update_threatened_pieces(self) -> None:
        # TODO Volgens mij is dit toch meer iets voor de evaluatie, omdat je de checkmate ook al in de get_legal_actions kan doen.
        # Clear the current list of threatened pieces for both players
        self.threatened_pieces_1.clear()
        self.threatened_pieces_2.clear()

        # Reset the check flag
        self.check = False

        # Iterate over both players
        for player in range(1, 3):
            # Identify the opposing player
            opposing_player: cython.int = 2 if player == 1 else 1

            # Get all legal actions for the opposing player
            opposing_moves: cython.list = self.get_legal_actions(
                self.board, opposing_player
            )

            # While we have generated the moves, we might as well store them.
            if opposing_player == self.player:
                self.current_player_moves = opposing_moves

            for move in opposing_moves:
                # Extract the destination position from the move
                dest_row: cython.int = move[2]
                dest_col: cython.int = move[3]

                # Get the piece at the destination
                target_piece: cython.int = self.board[dest_row][dest_col]

                # Check if the destination contains a piece of the current player
                if (player == 1 and 10 <= target_piece <= 18) or (
                    player == 2 and 1 <= target_piece <= 9
                ):
                    # Add the threatened piece to the list
                    if player == 1:
                        self.threatened_pieces_1.append(
                            (dest_row, dest_col, target_piece)
                        )
                    else:
                        self.threatened_pieces_2.append(
                            (dest_row, dest_col, target_piece)
                        )

                    # Check if the threatened piece is a king
                    if (
                        target_piece == PIECES["king"]
                        or target_piece == PIECES["king"] + 9
                    ):
                        # Set the check flag only if the current player's king is threatened
                        if self.player == player:
                            self.check = True
                        else:
                            assert (
                                False
                            ), "Opponent's king is threatened, this should not be possible."

    @cython.ccall
    def get_legal_actions(self) -> cython.list:
        legal_actions = []
        player_piece_start, player_piece_end = (1, 9) if self.player == 1 else (10, 18)
        for row in range(5):
            for col in range(5):
                piece = self.board[row][col]
                if (
                    player_piece_start <= piece <= player_piece_end
                ):  # Piece belongs to the current player
                    # Generate moves for this piece
                    moves = self.get_moves(row, col, piece)
                    legal_actions.extend(moves)

        # TODO Je moet hier kijken of je in check staat en of je uit check kan komen door een stuk te droppen.
        # TODO Je hebt daarvoor een extra methode nodig omdat simulate_move_exposes_king niet werkt voor drops.
        # TODO Je moet ook nog de is_legal_drop methode implementeren en checken of een drop wel mag op een bepaalde plek.
        # Drop actions
        captured_pieces = (
            self.captured_pieces_1 if self.player == 1 else self.captured_pieces_2
        )
        for piece in captured_pieces:
            for row in range(5):
                for col in range(5):
                    if self.board[row][col] == 0:  # Check for empty square
                        # Additional rules for specific pieces
                        if piece in [6, 15]:  # Pawn
                            # Check for pawn-specific rules (no two unpromoted pawns in the same column, not in the last row, etc.)
                            if row != 4:
                                legal_actions.append((-1, piece, row, col))
                        elif not (
                            piece in [6, 15] and row == 4
                        ):  # Other pieces, exclude last row for pawns
                            legal_actions.append((-1, piece, row, col))

        # If there are no legal actions, then we lost
        if len(legal_actions) == 0:
            self.winner = 3 - self.player
        else:
            # TODO Draws bepalen
            self.winner = -1

        return legal_actions

    @cython.ccall
    def get_moves(self, row: cython.int, col: cython.int) -> cython.list:
        moves = []
        piece = self.board[row][col]

        player_piece_start, player_piece_end = (1, 9) if piece <= 9 else (10, 18)

        for move in MOVES.get(piece, []):
            new_row, new_col = row, col
            while True:
                new_row += move[0]
                new_col += move[1]

                if not (0 <= new_row < 5 and 0 <= new_col < 5):
                    break  # Break if out of board bounds

                if self.board[new_row][new_col] != 0:
                    if (
                        player_piece_start
                        <= self.board[new_row][new_col]
                        <= player_piece_end
                    ):
                        break  # Break if encounter own piece
                    elif not self.simulate_move_exposes_king(
                        row, col, new_row, new_col
                    ):
                        moves.append(
                            (row, col, new_row, new_col)
                        )  # Capture opponent's piece
                    break

                if not self.simulate_move_exposes_king(row, col, new_row, new_col):
                    moves.append((row, col, new_row, new_col))

                # Stop extending the move for non-rook/bishop pieces
                if piece not in [
                    PIECES["rook"],
                    PIECES["bishop"],
                    PIECES["rook"] + 9,
                    PIECES["bishop"] + 9,
                ]:
                    break

        return moves

    @cython.ccall
    def king_under_attack_from_position(
        self,
        row: cython.int,
        col: cython.int,
        king_pos: cython.tuple[cython.int, cython.int],
    ) -> bool:
        piece = self.board[row][col]
        if piece not in MOVES:
            return False

        for move in MOVES[piece]:
            new_row, new_col = row, col
            while True:
                new_row += move[0]
                new_col += move[1]
                if not (0 <= new_row < 5 and 0 <= new_col < 5):
                    break  # Break if out of board bounds
                if (new_row, new_col) == king_pos:
                    return True  # Can attack the king
                # For pieces with limited movement
                if piece not in [
                    PIECES["rook"],
                    PIECES["bishop"],
                    PIECES["rook"] + 9,
                    PIECES["bishop"] + 9,
                ]:
                    break
                if self.board[new_row][new_col] != 0:
                    break  # Stop if any piece is in the way
        return False

    @cython.ccall
    def simulate_move_exposes_king(self, from_row, from_col, to_row, to_col) -> bool:
        # Temporarily make the move
        piece = self.board[from_row][from_col]
        captured_piece = self.board[to_row][to_col]
        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = 0

        # Determine the player whose king might be exposed
        player = 1 if piece <= 9 else 2

        # Check if the king is under attack after the move
        exposes_king = False
        king_pos = self.king_1 if player == 1 else self.king_2
        # Generate moves for all pieces of the opposing player and check if any can attack the king
        opposing_player_piece_start, opposing_player_piece_end = (
            (10, 18) if player == 1 else (1, 9)
        )
        for row in range(5):
            for col in range(5):
                piece = self.board[row][col]
                if opposing_player_piece_start <= piece <= opposing_player_piece_end:
                    if self.king_under_attack_from_position(row, col, king_pos):
                        exposes_king = True  # The king is under attack
                        break

            if exposes_king:
                break

        # Undo the move
        self.board[from_row][from_col] = piece
        self.board[to_row][to_col] = captured_piece

        return exposes_king

    @cython.ccall
    def valid_drop(self, piece, row, col) -> bool:
        # Check if the square is empty
        if self.board[row][col] != 0:
            return False

        # Pawn-specific rules
        if piece in [6, 15]:  # Pawn for both players
            # Check for another non-promoted Pawn in the same column
            if any(self.board[r][col] in [6, 15] for r in range(5)):
                return False
            # Pawns cannot be dropped on the last rank
            if row == 4:
                return False
            # Check for immediate checkmate by Pawn drop
            if self.does_pawn_drop_checkmate(piece, row, col):
                return False

        # Lance and Knight specific rules
        if piece in [7, 16]:  # Lance for both players
            if row == 4:  # Lance cannot be dropped on the last rank
                return False
        if piece in [8, 17]:  # Knight for both players
            if row >= 3:  # Knight cannot be dropped on the last two ranks
                return False

        return True

    @cython.ccall
    def does_pawn_drop_checkmate(self, piece, row, col) -> bool:
        # Temporarily drop the Pawn
        self.board[row][col] = piece

        # Check if this results in a checkmate
        opponent = 2 if self.player == 1 else 1
        in_checkmate = (
            not any(
                self.move_resolves_check(-1, -1, r, c)
                for r in range(5)
                for c in range(5)
            )
            and self.is_king_in_check()
        )

        # Remove the Pawn after checking
        self.board[row][col] = 0

        return in_checkmate

    @cython.ccall
    def check_if_in_check(self, action) -> bool:
        # Save the current state before making the move
        original_piece = self.board[to_row][to_col]
        moved_piece = self.board[from_row][from_col]

        # Make the move
        self.board[to_row][to_col] = moved_piece
        self.board[from_row][from_col] = 0

        # Check if the king is in check after the move
        in_check = self.is_in_check()

        # Undo the move
        self.board[from_row][from_col] = moved_piece
        self.board[to_row][to_col] = original_piece

        return not in_check

    @cython.ccall
    @cython.inline
    def promote_piece(self, row: cython.int, col: cython.int) -> None:
        if 2 <= self.board[row][col] <= 4:  # Rook, Bishop, Pawn
            # Promote the piece
            self.board[row][col] += 5

    @cython.ccall
    @cython.inline
    @cython.exceptval(-1)
    def get_demoted_piece(self, piece: cython.int) -> cython.int:
        if 2 <= piece <= 4:  # Rook, Bishop, Pawn
            # Promote the piece
            return piece - 5

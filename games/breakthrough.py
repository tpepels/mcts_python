# cython: language_level=3
import cython
import numpy as np
from cython.cimports import numpy as cnp

cnp.import_array()

from games.gamestate import normalize, win, loss, draw
import random
from cython.cimports.ai.c_random import c_shuffle, c_random

if cython.compiled:
    print("Breakthrough is compiled.")
else:
    print("Breakthrough is just a lowly interpreted script.")


# TODO Move this function elsewhere after optimisation
@cython.ccall
@cython.inline
@cython.profile(False)
@cython.infer_types(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.boundscheck(False)  # Turn off bounds-checking for entire function
@cython.wraparound(False)  # Turn off negative index wrapping for entire function
def where_is_k(board: cython.int[:], k: cython.int) -> cython.list:
    i: cython.int
    n: cython.int = board.shape[0]
    indices: cython.list = []

    for i in range(n):
        if board[i] == k:
            indices.append(i)

    return indices


dirs: cython.tuple[cython.int, cython.int, cython.int] = (-1, 0, 1)


@cython.cclass
@cython.initializedcheck(False)
@cython.nonecheck(False)
class BreakthroughGameState:
    zobrist_table = [[[random.randint(1, 2**61 - 1) for _ in range(3)] for _ in range(8)] for _ in range(8)]

    player = cython.declare(cython.int, visibility="public")
    board = cython.declare(cython.int[:], visibility="public")
    board_hash = cython.declare(cython.longlong, visibility="public")
    positions = cython.declare(cython.tuple[cython.list, cython.list], visibility="public")
    last_action = cython.declare(cython.tuple[cython.int, cython.int], visibility="public")

    winner: cython.int
    n_pieces_1: cython.int
    n_pieces_2: cython.int

    """
    This class represents the game state for the Breakthrough board game.
    Breakthrough is a two-player game played on an 8x8 board.
    Each player starts with 16 pieces occupying the first two rows.
    The goal is to move one of your pieces to the opponent's home row.
    """

    def __init__(
        self,
        board=None,
        player=1,
        board_hash=None,
        positions=None,
        winner=0,
        n_pieces_1=None,
        n_pieces_2=None,
        last_action=None,  # TODO Dit heb je toegevoegd
    ):
        """
        Initialize the game state with the given board configuration.
        If no board is provided, the game starts with the default setup.

        :param board: An optional board configuration.
        :param player: The player whose turn it is (1 or 2).
        """
        self.player = player
        self.board = board if board is not None else self._init_board()

        # Keep track of the pieces on the board, this keeps evaluation and move generation from recomputing these over and over
        if positions is None:
            self.positions = where_is_k(self.board, 1), where_is_k(self.board, 2)
        else:
            self.positions = positions
        if n_pieces_1 is None and n_pieces_2 is None:
            self.n_pieces_1 = 16
            self.n_pieces_2 = 16
        else:
            self.n_pieces_1 = n_pieces_1
            self.n_pieces_2 = n_pieces_2

        if last_action is None:
            self.last_action = (-1, -1)
        else:
            self.last_action = last_action  # The last action performed

        self.winner = winner
        self.board_hash = board_hash if board_hash is not None else self._init_hash()

    def _init_board(self):
        board = np.zeros(64, dtype=np.int32)
        board[:16] = 2
        board[48:] = 1
        return board

    def _init_hash(self):
        board_hash = 0
        for position in range(64):
            player = self.board[position]
            row, col = divmod(position, 8)
            board_hash ^= BreakthroughGameState.zobrist_table[row][col][player]
        return board_hash

    @cython.ccall
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.locals(
        caputured_player=cython.int,
    )
    def apply_action_playout(self, action: tuple[cython.int, cython.int]) -> BreakthroughGameState:
        """
        Apply the given action to create a new game state. The current state is not altered by this method.
        Actions are represented as a tuple (from_position, to_position).

        :param action: The action to apply.
        :return: A new game state with the action applied.
        """
        # TODO Deze heb je toegevoegd
        self.board[action[0]] = 0  # Remove the piece from its current position
        captured_player = self.board[action[1]]  # Check for captures
        self.board[action[1]] = self.player  # Place the piece at its new position

        # Keep track of the pieces on the board
        if captured_player != 0:
            self.positions[captured_player - 1].remove(action[1])
            # Keep track of the number of pieces of each player
            if captured_player == 2:
                self.n_pieces_2 -= 1
                if self.n_pieces_2 == 0:
                    self.winner = 1
            elif captured_player == 1:
                self.n_pieces_1 -= 1
                if self.n_pieces_1 == 0:
                    self.winner = 2

        self.positions[self.player - 1].remove(action[0])
        self.positions[self.player - 1].append(action[1])

        if self.winner == 0:
            if self.player == 1 and action[1] < 8:
                self.winner = 1
            elif self.player == 2 and action[1] >= 56:
                self.winner = 2

        self.player = 3 - self.player
        self.last_action = action

    @cython.ccall
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.locals(
        from_position=cython.int,
        to_position=cython.int,
        new_board=cython.int[:],
        player=cython.int,
        current_player=cython.int,
        caputured_player=cython.int,
        from_row=cython.int,
        to_row=cython.int,
        to_col=cython.int,
        from_col=cython.int,
        board_hash=cython.longlong,
    )
    def apply_action(self, action: tuple[cython.int, cython.int]) -> BreakthroughGameState:
        """
        Apply the given action to create a new game state. The current state is not altered by this method.
        Actions are represented as a tuple (from_position, to_position).

        :param action: The action to apply.
        :return: A new game state with the action applied.
        """
        new_board = self.board.copy()
        n_pieces_1: cython.int = self.n_pieces_1
        n_pieces_2: cython.int = self.n_pieces_2

        winner: cython.int = 0

        new_board[action[0]] = 0  # Remove the piece from its current position
        captured_player = new_board[action[1]]
        new_board[action[1]] = self.player  # Place the piece at its new position

        new_positions: cython.tuple[cython.list, cython.list] = (
            self.positions[0].copy(),
            self.positions[1].copy(),
        )
        # Keep track of the pieces on the board
        if captured_player != 0:
            new_positions[captured_player - 1].remove(action[1])
            # Keep track of the number of pieces of each player
            if captured_player == 2:
                n_pieces_2 -= 1
                if n_pieces_2 == 0:
                    winner = 1
            elif captured_player == 1:
                n_pieces_1 -= 1
                if n_pieces_1 == 0:
                    winner = 2

        new_positions[self.player - 1].remove(action[0])
        new_positions[self.player - 1].append(action[1])

        from_row = action[0] // 8
        from_col = action[0] % 8
        to_row = action[1] // 8
        to_col = action[1] % 8

        board_hash = (
            self.board_hash
            ^ BreakthroughGameState.zobrist_table[from_row][from_col][self.player]
            ^ BreakthroughGameState.zobrist_table[to_row][to_col][captured_player]
            ^ BreakthroughGameState.zobrist_table[to_row][to_col][self.player]
        )

        if winner == 0:
            if self.player == 1 and action[1] < 8:
                winner = 1
            elif self.player == 2 and action[1] >= 56:
                winner = 2

        return BreakthroughGameState(
            new_board,
            3 - self.player,
            board_hash=board_hash,
            positions=new_positions,
            winner=winner,
            n_pieces_1=n_pieces_1,
            n_pieces_2=n_pieces_2,
            last_action=action,
        )

    @cython.ccall
    def skip_turn(self):
        """Used for the null-move heuristic in alpha-beta search

        Returns:
            BreakthroughGameState: A new gamestate in which the players are switched but no move performed
        """
        # Pass the same board hash since this is only used for null moves
        return BreakthroughGameState(self.board.copy(), 3 - self.player, board_hash=self.board_hash)

    @cython.ccall
    @cython.returns(tuple[cython.int, cython.int])
    def get_random_action(self):
        """
        Generate a single legal action for the current player.

        :return: A tuple representing a legal action (from_position, to_position). If there are no legal actions, returns None.
        """

        dr: cython.int = -1
        if self.player == 2:
            dr = 1
        positions: cython.list = self.positions[self.player - 1]

        n: cython.int = len(positions)
        start: cython.int = c_random(0, n - 1)  # This allows us to start at a random piece

        i: cython.int

        safe_captures: cython.list = []
        all_moves: cython.list = []
        captures: cython.list = []
        for i in range(n):
            index: cython.int = (start + i) % n
            position: cython.int = positions[index]

            row: cython.int = position // 8
            col: cython.int = position % 8

            start_dc: cython.int = c_random(0, 2)  # This allows us to start at in a random direction
            k: cython.int

            for k in range(3):
                dc: cython.int = dirs[(start_dc + k) % 3]
                new_row: cython.int = row + dr
                new_col: cython.int = col + dc
                if not (
                    (0 <= new_row) & (new_row < 8) & (0 <= new_col) & (new_col < 8)
                ):  # if the new position is not in bounds, skip to the next direction
                    continue
                new_position: cython.int = new_row * 8 + new_col

                # Straight, no capture or diagonal capture / empty cell
                if (dc == 0 and self.board[new_position] == 0) or (self.board[new_position] != self.player):
                    # Decisive moves
                    if (self.player == 1 and new_position < 8) or (self.player == 2 and new_position >= 56):
                        return (position, new_position)

                    # Captures
                    if new_position == (3 - self.player):
                        # Anti-decisive moves (captures from the last row)
                        if (self.player == 1 and position < 56) or (self.player == 2 and position < 8):
                            return (position, new_position)

                        # Prioritize safe captures
                        if is_safe(position, self.player, self.board):
                            safe_captures.append((position, new_position))
                        else:
                            # give captures higher chance of being selected
                            captures.append((position, new_position))

                    # Safe captures are prioritized anyway so no need to add non-captures
                    if safe_captures == [] and captures == []:
                        all_moves.append((position, new_position))

        # Always do a safe capture if you can
        if safe_captures != []:
            if len(safe_captures) > 1:
                return safe_captures[c_random(0, len(safe_captures) - 1)]
            return safe_captures[0]
        if captures != []:
            if len(captures) > 1:
                return captures[c_random(0, len(captures) - 1)]
            return captures[0]
        # All_moves includes captures, they'll be selected with a higher probability
        return all_moves[c_random(0, len(all_moves) - 1)]

    @cython.ccall
    def get_legal_actions(self) -> cython.list:
        """
        Get all legal actions for the current player.

        :return: A list of legal actions as tuples (from_position, to_position).
        In case of a terminal state, an empty list is returned.
        """
        legal_actions: cython.list = []
        dr: cython.int = -1
        if self.player == 2:
            dr = 1
        i: cython.int
        for i in range(len(self.positions[self.player - 1])):
            position: cython.int = self.positions[self.player - 1][i]
            row: cython.int = position // 8
            col: cython.int = position % 8
            dc: cython.int
            for dc in range(-1, 2):
                new_row: cython.int = row + dr
                new_col: cython.int = col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:  # Check if the new position is within the board
                    new_position: cython.int = new_row * 8 + new_col

                    if dc == 0:  # moving straight
                        if self.board[new_position] == 0:
                            legal_actions.append((position, new_position))

                    else:  # diagonal capture / move
                        if self.board[new_position] != self.player:
                            legal_actions.append((position, new_position))
        return legal_actions

    @cython.ccall
    def is_terminal(self) -> cython.bint:
        """
        Check if the current game state is terminal (i.e., a player has won).

        :return: True if the game state is terminal, False otherwise.
        """
        if self.winner != 0:
            return 1
        return 0

    @cython.ccall
    def get_reward(self, player: cython.int) -> cython.int:
        if self.winner == player:
            return win
        elif self.winner == 3 - player:
            return loss

        return draw

    @cython.ccall
    @cython.returns(cython.tuple[cython.double, cython.double])
    def get_result_tuple(self):
        # TODO Deze heb je toegevoegd
        if self.winner == 1:
            return (1.0, 0.0)
        elif self.winner == 2:
            return (0.0, 1.0)

        return (0.5, 0.5)

    @cython.cfunc
    def is_capture(self, move: cython.tuple[cython.int, cython.int]):
        """
        Check if the given move results in a capture.

        :param move: The move to check as a tuple (from_position, to_position).
        :return: True if the move results in a capture, False otherwise.
        """
        # Check if the destination cell contains an opponent's piece
        return self.board[move[1]] == 3 - self.player

    @cython.ccall
    def evaluate_moves(self, moves: cython.list):
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

    @cython.ccall
    def move_weights(self, moves: cython.list):
        scores: cython.list = [0] * len(moves)
        i: cython.int
        for i in range(len(moves)):
            scores[i] = self.evaluate_move(moves[i])
        return scores

    @cython.ccall
    def evaluate_move(self, move: cython.tuple[cython.int, cython.int]) -> cython.int:
        """
        Evaluate the given move using a simple heuristic: each step forward is worth 1 point,
        and capturing an opponent's piece is worth 2 points.

        :param move: The move to evaluate.
        :return: The heuristic value of the move.
        """
        score: cython.int = 0

        # Use lorentz_values for base_value, the max value is 36, so we add 36 to the score to make sure it is non-negative
        if self.player == 2:
            score = 36 + (lorentz_values[move[1]] - lorentz_values[move[0]])
        else:
            # Player 1 views the lorenz_values in reverse
            score = 36 + (lorentz_values[63 - move[1]] - lorentz_values[63 - move[0]])

        # Reward capturing
        if self.board[move[1]] == 3 - self.player:
            score = int(score**2)  # square score if it's a capture
            # An antidecisive move
            if (self.player == 1 and move[0] >= 56) or (self.player == 2 and move[0] < 8):
                score += 1000000  # Add a very high score if the move is antidecisive

        # Reward safe positions
        if is_safe(move[1], self.player, self.board):
            score *= 2  # Add base_value again if the position is safe

        # Reward decisive moves
        # The decisive condition checks for a piece on the penultimate row that can move to the opponent's final row without the opportunity of being captured.
        if (self.player == 1 and (8 <= move[0] <= 16)) or (self.player == 2 and (48 <= move[0] <= 56)):
            score += 1000000  # Add a very high score if the move is decisive

        return score

    def visualize(self, full_debug=False):
        """
        Visualize the board for the Breakthrough game.
        """
        output = ""
        cell_representation = {0: ".", 1: "W", 2: "B"}
        column_letters = "  " + " ".join("ABCDEFGH") + "\n"  # Use chess-style notation

        for i in range(8):
            row = [cell_representation.get(piece, ".") for piece in self.board[i * 8 : i * 8 + 8]]
            formatted_row = " ".join(row)
            output += f"{i + 1} {formatted_row}\n"

        output += column_letters
        output += f"Player: {self.player}\n"

        if full_debug:
            if not self.validate_actions():
                print(" !!! Invalid actions detected !!! ")

            output += "Hash: " + str(self.board_hash) + "\n"
            output += f"Reward: {self.get_reward(1)}/{self.get_reward(2)}, Terminal: {self.is_terminal()}\n"

            moves = self.get_legal_actions()
            output += f"{len(moves)} actions: {','.join([self.readable_move(x) for x in moves])}\n"
            # ---------------------------Evaluations---------------------------------
            output += f"Simple evaluation: {evaluate_breakthrough(self, 1, norm=False)}/{evaluate_breakthrough(self, 2, norm=False)}\n"
            output += f"Lorenz evaluation: {evaluate_breakthrough_lorenz(self, 1, norm=False)}/{evaluate_breakthrough_lorenz(self, 2, norm=False)}\n"
            output += f"Normalized:"
            output += f"Simple evaluation: {evaluate_breakthrough(self, 1, norm=True)}/{evaluate_breakthrough(self, 2, norm=True)}\n"
            output += f"Lorenz evaluation: {evaluate_breakthrough_lorenz(self, 1, norm=True)}/{evaluate_breakthrough_lorenz(self, 2, norm=True)}\n"

            white_pieces = self.positions[0]
            black_pieces = self.positions[1]

            output += f"# white pieces: {len(white_pieces)} | # black pieces: {len(black_pieces)}\n"
            # ---------------------------Endgame-------------------------------------------
            output += f"is endgame: {np.count_nonzero(self.board) < 16}\n"
            output += "..." * 60 + "\n"
            # ---------------------------Possible captures---------------------------------
            wh_caps = count_capture_moves(self.board, white_pieces, 1)
            bl_caps = count_capture_moves(self.board, black_pieces, 2)
            output += f"White has {wh_caps} capture value, black has {bl_caps}.\n"
            output += "..." * 60 + "\n"
            # ---------------------------Piece safety--------------------------------------
            for piece in black_pieces:
                if not is_safe(piece, 2, self.board):
                    output += f"{self.readable_location(piece)} - black is not safe\n"

            for piece in white_pieces:
                if not is_safe(piece, 1, self.board):
                    output += f"{self.readable_location(piece)} - white is not safe\n"
            output += "..." * 60 + "\n"
            # ---------------------------Decisiveness---------------------------------
            p1_decisive, p2_decisive = is_decisive(self)
            output += f"Player 1 decisive: {p1_decisive}, Player 2 decisive: {p2_decisive}\n"
            output += "..." * 60 + "\n"

            if len(moves) > 0:
                actions = self.evaluate_moves(self.get_legal_actions())
                actions = sorted(actions, key=lambda x: x[1], reverse=True)
                output += "..." * 60 + "\n"
                output += str([(self.readable_move(a[0]), a[1]) for a in actions]) + "\n"

            output += "..." * 60 + "\n"
            # ---------------------------Piece mobility---------------------------------
            for piece in black_pieces:
                output += f"{self.readable_location(piece)} - black piece mobility: {piece_mobility(piece, 2, self.board)}\n"
            for piece in white_pieces:
                output += f"{self.readable_location(piece)} - white piece mobility: {piece_mobility(piece, 1, self.board)}\n"

        return output

    def validate_actions(self):
        """
        Check if the available actions are valid or not.

        :return: A boolean indicating whether all actions are valid or not.
        """
        actions = self.get_legal_actions()
        # Make sure all actions are unique.
        if len(actions) != len(set(actions)):
            print("Found duplicate actions.")
            return False

        for action in actions:
            from_position, to_position = action

            # Check that there is a piece belonging to the current player at the 'from' position.
            if self.board[from_position] != self.player:
                print(f"No piece for current player at {to_chess_notation(from_position)}.")
                return False

            # Check that there is not a piece belonging to the current player at the 'to' position.
            if self.board[to_position] == self.player:
                print(f"Existing piece for current player at {to_chess_notation(to_position)}.")
                return False

            # Check that all moves are in the correct direction (higher rows for p2, lower rows for p1).
            if self.player == 1 and to_position >= from_position:
                print(f"Invalid move direction for player 1 at {self.readable_move(action)}.")
                return False
            elif self.player == 2 and to_position <= from_position:
                print(f"Invalid move direction for player 2 at {self.readable_move(action)}.")
                return False

        return True

    def readable_move(self, move):
        """
        returns a move in a more human-friendly format.

        :param move: A tuple (from_position, to_position) in board-index style.
        """

        from_position, to_position = move
        return f"{to_chess_notation(from_position)} -> {to_chess_notation(to_position)}"

    def readable_location(self, position):
        return to_chess_notation(position)

    @property
    def transposition_table_size(self):
        # return an appropriate size based on the game characteristics
        return 2**24

    def __repr__(self) -> str:
        return "breakthrough"


def to_chess_notation(index):
    """Transform a board index into chess notation."""
    row, col = divmod(index, 8)
    return f"{chr(col + 65)}{row + 1}"


@cython.ccall
@cython.infer_types(True)
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def evaluate_breakthrough(
    state: BreakthroughGameState,
    player: cython.int,
    m_piece: cython.double = 1.0,
    m_dist: cython.double = 1.0,
    m_near: cython.double = 1.0,
    m_blocked: cython.double = 0.25,
    m_opp_disc: cython.double = 0.9,
    a: cython.int = 20,
    norm: cython.bint = 0,
) -> cython.double:
    """
    Evaluates the current game state for Breakthrough using a custom evaluation function,
    which considers number of pieces, average distance of pieces to the opponent's side,
    number of pieces close to the opponent's side, and number of blocked pieces.

    :param state: The game state to evaluate.
    :param player: The player to evaluate for (1 or 2).
    :param m_piece: Weight assigned to the difference in number of pieces.
    :param m_dist: Weight assigned to the difference in average distance to opponent's side.
    :param m_near: Weight assigned to the difference in number of pieces close to opponent's side.
    :param m_blocked: Weight assigned to the difference in number of blocked pieces.
    :param m_opp_disc: Multiplier for the evaluation score if it's the opponent's turn.
    :param a: Normalization factor for the evaluation score.
    :param norm: If True, the function will return a normalized evaluation score.

    :return: The evaluation score for the given player.
    """

    player_pieces: cython.int = 0
    player_distance: cython.int = 0
    player_near_opponent_side: cython.int = 7
    player_blocked: cython.int = 0

    opponent_pieces: cython.int = 0
    opponent_distance: cython.int = 0
    opponent_near_opponent_side: cython.int = 7
    opponent_blocked: cython.int = 0
    board: cython.int[:] = state.board
    positions: cython.list = state.positions[0] + state.positions[1]
    new_position: cython.int
    new_col: cython.int
    new_row: cython.int
    y: cython.int
    x: cython.int
    piece: cython.int
    dr: cython.int
    dc: cython.int

    for i in range(len(positions)):
        piece = board[positions[i]]
        x = positions[i] // 8
        y = positions[i] % 8

        if piece == player:
            player_pieces += 1
            dist = 7 - x if piece == 2 else x
            player_distance += dist

            if (piece == 2 and x >= 4) or (piece == 1 and x <= 3):
                player_near_opponent_side = min(dist, player_near_opponent_side)

        else:
            opponent_pieces += 1
            dist = 7 - x if piece == 2 else x
            opponent_distance += dist

            if (piece == 2 and x >= 4) or (piece == 1 and x <= 3):
                opponent_near_opponent_side = min(dist, opponent_near_opponent_side)

        # Determine the direction of movement based on the current player
        if piece == 1:
            dr = -1
        else:
            dr = 1

        blocked = True
        for dc in range(-1, 2):
            new_row = x + dr
            new_col = y + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                new_position = new_row * 8 + new_col
                if board[new_position] == 0 or (dc != 0 and board[new_position] != piece):
                    blocked = False
                    break

        if piece == player:
            player_blocked += 1 if blocked else 0
        else:
            opponent_blocked += 1 if blocked else 0

    eval_value = m_piece * (player_pieces - opponent_pieces)
    eval_value += m_dist * (opponent_distance - player_distance)
    eval_value += m_near * (player_near_opponent_side - opponent_near_opponent_side)
    eval_value += m_blocked * (opponent_blocked - player_blocked)

    if state.player != player:
        eval_value *= m_opp_disc

    if norm:
        return normalize(eval_value, a)
    else:
        return eval_value


@cython.ccall
@cython.infer_types(True)
def evaluate_breakthrough_lorenz(
    state: BreakthroughGameState,
    player: cython.int,
    m_lorenz: cython.double = 20.0,
    m_mobility: cython.double = 2.0,
    m_safe: cython.double = 15.0,
    m_endgame: cython.double = 11.0,
    m_cap: cython.double = 10.0,
    m_piece_diff: cython.double = 13.0,
    m_opp_disc: cython.double = 1.1,
    m_decisive: cython.double = 800.0,
    a: cython.int = 2000,
    norm: cython.bint = 0,
) -> cython.double:
    """
    Evaluates the current game state using an enhanced Lorenz evaluation function,
    which takes into account the positioning of pieces, their mobility, the safety of their positions,
    whether the piece is blocked, the phase of the endgame, and the potential to capture opposing pieces.

    :param state: The game state to evaluate.
    :param player: The player to evaluate for (1 or 2).
    :param m_lorenz: Weight assigned to the Lorenz value of the board configuration.
    :param m_mobility: Weight assigned to the mobility of the player's pieces.
    :param m_safe: Weight assigned to the safety of the player's pieces.
    :param m_endgame: Weight assigned to the endgame phase.
    :param m_cap: Weight assigned to the capture difference between the player and the opponent.
    :param m_cap_move: Weight assigned to the difference in potential capture moves between the player and the opponent.
    :param m_opp_disc: Multiplier for the evaluation score if it's the opponent's turn.
    :param a: Normalization factor for the evaluation score.
    :param norm: If True, the function will return a normalized evaluation score.

    :return: The evaluation score for the given player.
    """
    board_values: cython.double = 0
    mobility_values: cython.double = 0
    safety_values: cython.double = 0
    endgame_values: cython.double = 0
    piece_diff: cython.double = 0
    pieces: cython.int = 0
    decisive_values: cython.double = 0
    caps: cython.double = 0
    opponent: cython.int = 3 - player
    board: cython.int[:] = state.board
    positions: cython.list = state.positions[0] + state.positions[1]
    multiplier: cython.double
    piece_value: cython.double
    mob_val: cython.double
    position: cython.int
    piece: cython.int

    for i in range(len(positions)):
        position = positions[i]
        piece = board[position]
        # Player or opponent
        if piece == player:
            multiplier = 1.0
        else:
            multiplier = -1.0

        pieces += 1
        piece_diff += multiplier
        if piece == 2:
            piece_value = lorentz_values[position]
        else:
            piece_value = lorentz_values[63 - position]

        board_values += multiplier * piece_value

        if m_mobility != 0.0:
            mob_val = piece_mobility(position, piece, board)
            mobility_values += multiplier * mob_val * (1 + piece_value)

        if m_safe != 0.0 and is_safe(position, piece, board):
            safety_values += multiplier * piece_value

    if m_endgame != 0.0 and pieces < 12:
        endgame_values = m_endgame * piece_diff

    if m_decisive != 0.0:
        player_1_decisive, player_2_decisive = is_decisive(state)

        if player == 1:
            decisive_values = m_decisive if player_1_decisive else 0
            decisive_values -= m_decisive if player_2_decisive else 0
        else:  # self.player == 2
            decisive_values = m_decisive if player_2_decisive else 0
            decisive_values -= m_decisive if player_1_decisive else 0

    if m_cap >= 0:
        caps = count_capture_moves(board, state.positions[player - 1], player) - count_capture_moves(
            board, state.positions[opponent - 1], opponent
        )

    eval_value: cython.double = (
        decisive_values
        + endgame_values
        + m_lorenz * board_values
        + m_mobility * mobility_values
        + m_safe * safety_values
        + m_piece_diff * piece_diff
        + m_cap * caps
    )

    if state.player == opponent:
        eval_value *= m_opp_disc

    if norm:
        return normalize(eval_value, a)
    else:
        return eval_value


# indices of the penultimate row for player 1
bef_last_rows_2: cnp.ndarray = np.arange(48, 56)
# indices of the penultimate row for player 2
bef_last_rows_1: cnp.ndarray = np.arange(8, 16)


@cython.cfunc
@cython.locals(
    player_1_decisive=cython.int,
    player_2_decisive=cython.int,
    pos=cython.int,
    i=cython.int,
)
@cython.returns(tuple[cython.int, cython.int])
def is_decisive(state: BreakthroughGameState):
    player_1_decisive = player_2_decisive = 0

    for i in range(len(state.positions[0])):
        pos = state.positions[0][i]
        # Decisive moves
        if 8 <= pos < 16:
            player_1_decisive += 1

    for i in range(len(state.positions[1])):
        pos = state.positions[1][i]
        # Decisive moves
        if 48 < pos <= 56:
            player_2_decisive += 1

    return player_1_decisive, player_2_decisive


BL_DIR: cython.int[:, :] = np.array([[1, 0], [1, -1], [1, 1]], dtype=np.dtype("i"))
WH_DIR: cython.int[:, :] = np.array([[-1, 0], [-1, -1], [-1, 1]], dtype=np.dtype("i"))


@cython.cfunc
@cython.infer_types(True)
@cython.boundscheck(False)
@cython.locals(
    position=cython.int,
    player=cython.int,
    board=cython.int[:],
    row=cython.int,
    col=cython.int,
    mobility=cython.int,
    directions=cython.int[:, :],
    dr=cython.int,
    dc=cython.int,
    new_row=cython.int,
    new_col=cython.int,
    new_position=cython.int,
)
def piece_mobility(position, player, board) -> cython.int:
    """
    Calculates the mobility of a piece at the given position.

    :param position: The position of the piece on the board.
    :param player: The player to which the piece belongs (1 or 2).
    :param board: The game state.
    :return: The mobility value of the piece.
    """
    row = position // 8
    col = position % 8
    mobility = 0

    # The directions of the moves depend on the player.
    # White (1) moves up (to lower rows), Black (2) moves down (to higher rows).
    if player == 1:  # White
        directions = WH_DIR  # Forward, and diagonally left and right
    else:  # Black
        directions = BL_DIR  # Forward, and diagonally left and right

    for i in range(3):
        dr = directions[i, 0]
        dc = directions[i, 1]
        new_row = row + dr
        new_col = col + dc
        if 0 <= new_row < 8 and 0 <= new_col < 8:
            new_position = new_row * 8 + new_col
            if (dc == 0 and board[new_position] == 0) or (
                abs(dc) == 1 and (board[new_position] == 0 or board[new_position] != player)
            ):
                mobility += 1

    return mobility


@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
@cython.locals(
    position=cython.int,
    player=cython.int,
    board=cython.int[:],
    x=cython.int,
    y=cython.int,
    row_direction=cython.int,
    col_direction=cython.int,
    attackers=cython.int,
    defenders=cython.int,
    opponent=cython.int,
    i=cython.int,
    new_x=cython.int,
    new_y=cython.int,
    new_position=cython.int,
    piece=cython.int,
)
def is_safe(position, player, board) -> cython.bint:
    """
    Determines if a piece at a given position is safe from being captured.

    :param position: The position of the piece.
    :param player: The player to which the piece belongs (1 or 2).
    :param board: The game board.
    :return: True if the piece is safe, False otherwise.
    """

    # Convert linear position to 2D grid coordinates.
    x = position // 8
    y = position % 8

    # Initialize counters for opponent pieces (attackers) and own pieces (defenders) around the current piece.
    attackers = 0
    defenders = 0

    # Determine opponent's player number
    opponent = 3 - player

    # Define directions for row and column
    row_direction = -1
    if player == 2:
        row_direction = 1
    col_direction = -1  # start with -1 direction, then change to 1 in the loop

    # For each direction (-1, 1), check the positions in the row direction (straight and diagonals).
    for _ in range(2):
        # Check the first row direction (upwards for player 1, downwards for player 2).
        new_x = x + row_direction
        new_y = y + col_direction

        # If the new position is within the board boundaries
        if 0 <= new_x < 8 and 0 <= new_y < 8:
            # Calculate linear position from grid coordinates.
            new_position = new_x * 8 + new_y
            piece = board[new_position]

            # If there's an opponent piece in this position, increment the attackers counter.
            if piece == opponent:
                attackers += 1

        # Check the second row direction (downwards for player 1, upwards for player 2).
        new_x = x - row_direction
        new_y = y + col_direction

        # If the new position is within the board boundaries
        if 0 <= new_x < 8 and 0 <= new_y < 8:
            # Calculate linear position from grid coordinates.
            new_position = new_x * 8 + new_y
            piece = board[new_position]

            # If there's a friendly piece in this position, increment the defenders counter.
            if piece == player:
                defenders += 1

        # Change the column direction for the second iteration
        col_direction = 1
    # The piece is considered safe if the number of attackers is less than or equal to the number of defenders.
    return attackers <= defenders


@cython.cfunc
@cython.infer_types(True)
@cython.boundscheck(False)
@cython.locals(
    board=cython.int[:],
    positions=cython.list,
    player=cython.int,
    total_capture_moves=cython.int,
    position=cython.int,
    row=cython.int,
    col=cython.int,
    dr=cython.int,
    piece_capture_moves=cython.int,
    dc=cython.int,
    new_row=cython.int,
    new_col=cython.int,
    new_position=cython.int,
    i=cython.int,
    n=cython.int,
    opponent=cython.int,
    piece_value=cython.int,
)
def count_capture_moves(board, positions, player) -> cython.int:
    """Count the number of pieces that can capture and the total number of possible capture moves for a given player.

    :param game_state: The game state instance.
    :param player: The player (1 or 2) to count capture moves for.
    :return: The number of pieces that can capture and the total number of possible capture moves for the player.
    """
    total_capture_moves = 0
    opponent = 3 - player
    n: cython.int = len(positions)

    for i in range(n):
        position = positions[i]
        row = position // 8
        col = position % 8

        dr = -1
        if player == 2:
            dr = 1

        piece_capture_moves = 0

        for dc in range(-1, 2, 2):  # Only diagonal movements can result in capture
            new_row = row + dr
            new_col = col + dc

            if (
                0 <= new_row < 8 and 0 <= new_col < 8
            ):  # Check if the new position is within the board boundaries
                new_position = new_row * 8 + new_col
                # Check if the new position contains an opponent's piece
                if board[new_position] == opponent:
                    # Add the lorenz value from the opponent's perspective
                    # Assume that we'll capture the most valuable piece
                    piece_value = (
                        lorentz_values[new_position] if opponent == 2 else lorentz_values[63 - new_position]
                    )
                    piece_capture_moves = max(piece_capture_moves, piece_value)

        total_capture_moves += piece_capture_moves

    return int(total_capture_moves)


# List of values representing the importance of each square on the board. In view of player 2.
lorentz_values: cython.int[:] = np.array(
    [
        5,
        15,
        15,
        5,
        5,
        15,
        15,
        5,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        2,
        4,
        6,
        6,
        6,
        6,
        6,
        6,
        4,
        7,
        10,
        10,
        10,
        10,
        10,
        10,
        7,
        11,
        15,
        15,
        15,
        15,
        15,
        15,
        11,
        16,
        21,
        21,
        21,
        21,
        21,
        21,
        16,
        20,
        28,
        28,
        28,
        28,
        28,
        28,
        20,
        36,
        36,
        36,
        36,
        36,
        36,
        36,
        36,
    ],
    dtype=np.int32,
)

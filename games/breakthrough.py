# cython: language_level=3, initializedcheck=False

import cython
import numpy as np
from cython.cimports import numpy as cnp

cnp.import_array()

from games.gamestate import GameState, normalize, win, loss, draw
import random

if cython.compiled:
    print("Breakthrough is compiled.")
else:
    print("Breakthrough is just a lowly interpreted script.")


class BreakthroughGameState(GameState):
    players_bitstrings = [random.randint(1, 2**32 - 1) for _ in range(3)]  # 0 is for the empty player
    zobrist_table = [[[random.randint(1, 2**32 - 1) for _ in range(3)] for _ in range(8)] for _ in range(8)]

    """
    This class represents the game state for the Breakthrough board game.
    Breakthrough is a two-player game played on an 8x8 board.
    Each player starts with 16 pieces occupying the first two rows.
    The goal is to move one of your pieces to the opponent's home row.
    """

    def __init__(self, board=None, player=1, board_hash=None):
        """
        Initialize the game state with the given board configuration.
        If no board is provided, the game starts with the default setup.

        :param board: An optional board configuration.
        :param player: The player whose turn it is (1 or 2).
        """
        self.player = player
        self.board = board if board is not None else self._init_board()
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
            board_hash ^= self.zobrist_table[row][col][player]
        board_hash ^= self.players_bitstrings[self.player]
        return board_hash

    def apply_action(self, action):
        """
        Apply the given action to create a new game state. The current state is not altered by this method.
        Actions are represented as a tuple (from_position, to_position).

        :param action: The action to apply.
        :return: A new game state with the action applied.
        """
        from_position, to_position = action
        new_board = np.copy(self.board)

        player = new_board[from_position]
        new_board[from_position] = 0  # Remove the piece from its current position
        captured_player = new_board[to_position]
        new_board[to_position] = player  # Place the piece at its new position

        from_row, from_col = divmod(from_position, 8)
        to_row, to_col = divmod(to_position, 8)

        board_hash = (
            self.board_hash
            ^ self.zobrist_table[from_row][from_col][player]
            ^ self.zobrist_table[to_row][to_col][captured_player]
            ^ self.zobrist_table[to_row][to_col][player]
            ^ self.players_bitstrings[self.player]
            ^ self.players_bitstrings[3 - self.player]
        )

        return BreakthroughGameState(new_board, 3 - self.player, board_hash=board_hash)

    def skip_turn(self):
        """Used for the null-move heuristic in alpha-beta search

        Returns:
            BreakthroughGameState: A new gamestate in which the players are switched but no move performed
        """
        # Pass the same board hash since this is only used for null moves
        return BreakthroughGameState(np.copy(self.board), 3 - self.player, board_hash=self.board_hash)

    def get_random_action(self):
        """
        Generate a single legal action for the current player.

        :return: A tuple representing a legal action (from_position, to_position). If there are no legal actions, returns None.
        """
        return get_random_action(self.board, self.player)

    def yield_legal_actions(self):
        """
        Yield all legal actions for the current player.

        :yield: Legal actions as tuples (from_position, to_position). In case of a terminal state, an empty sequence is returned.
        """
        # TODO It may be faster to just generate all actions using the cython function..
        positions = np.where(self.board == self.player)[0]
        random.shuffle(positions)  # Shuffle positions

        dr = -1 if self.player == 1 else 1

        dc_values = [-1, 0, 1]
        for position in positions:
            row, col = divmod(position, 8)
            random.shuffle(dc_values)  # Shuffle dc_values for each position

            for dc in dc_values:
                new_row, new_col = row + dr, col + dc

                if 0 <= new_row < 8 and 0 <= new_col < 8:  # Check if new position is in bounds
                    new_position = new_row * 8 + new_col

                    if dc == 0 and self.board[new_position] == 0:  # moving straight
                        yield position, new_position
                    elif dc != 0 and self.board[new_position] != self.player:  # capturing / diagonal move
                        yield position, new_position

    def get_legal_actions(self):
        """
        Get all legal actions for the current player.

        :return: A list of legal actions as tuples (from_position, to_position). In case of a terminal state, an empty list is returned.
        """
        return get_legal_actions(self.player, self.board)

    def is_terminal(self):
        """
        Check if the current game state is terminal (i.e., a player has won).

        :return: True if the game state is terminal, False otherwise.
        """
        return (self.board[:8] == 1).any() or (self.board[56:] == 2).any()

    def get_reward(self, player):
        if (self.board[:8] == 1).any():
            return win if player == 1 else loss
        elif (self.board[56:] == 2).any():
            return win if player == 2 else loss
        else:
            return draw

    def is_capture(self, move):
        """
        Check if the given move results in a capture.

        :param move: The move to check as a tuple (from_position, to_position).
        :return: True if the move results in a capture, False otherwise.
        """
        # Check if the destination cell contains an opponent's piece
        return self.board[move[1]] == 3 - self.player

    def evaluate_moves(self, moves):
        """
        Evaluate the given moves using a simple heuristic: each step forward is worth 1 point,
        and capturing an opponent's piece is worth 2 points.

        :param moves: The list of moves to evaluate.
        :return: The list of heuristic values of the moves.
        """
        scores: list[tuple] = [()] * len(moves)
        for i in range(len(moves)):
            scores[i] = (moves[i], evaluate_move(self.board, moves[i][0], moves[i][1], self.player))
        return scores

    def move_weights(self, moves):
        scores = [0] * len(moves)
        for i in range(len(moves)):
            scores[i] = evaluate_move(self.board, moves[i][0], moves[i][1], self.player)
        return scores

    def evaluate_move(self, move):
        """
        Evaluate the given move using a simple heuristic: each step forward is worth 1 point,
        and capturing an opponent's piece is worth 2 points.

        :param move: The move to evaluate.
        :return: The heuristic value of the move.
        """

        return evaluate_move(self.board, move[0], move[1], self.player)

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

            white_pieces = np.where(self.board == 1)[0]
            black_pieces = np.where(self.board == 2)[0]

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
            p1_decisive, p2_decisive, p1_antidecisive, p2_antidecisive = is_decisive(self)
            output += f"Player 1 decisive: {p1_decisive}, Player 2 decisive: {p2_decisive}, Player 1 antidecisive: {p1_antidecisive}, Player 2 antidecisive: {p2_antidecisive}\n"
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
        return 2**19

    def __repr__(self) -> str:
        return "breakthrough"


def to_chess_notation(index):
    """Transform a board index into chess notation."""
    row, col = divmod(index, 8)
    return f"{chr(col + 65)}{row + 1}"


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.locals(
    positions=cython.long[:],
    position=cython.int,
    row=cython.int,
    i=cython.int,
    col=cython.int,
    dr=cython.int,
    dc=cython.int,
    ran=cython.list,
    new_row=cython.int,
    new_col=cython.int,
    in_bounds=cython.bint,
    new_position=cython.int,
    board=cnp.ndarray,
    player=cython.int,
)
def get_random_action(board, player) -> tuple[cython.int, cython.int]:
    """
    Generate a single legal action for the current player.

    :return: A tuple representing a legal action (from_position, to_position). If there are no legal actions, returns None.
    """
    positions = np.where(board == player)[0]
    random.shuffle(positions)  # shuffle the positions to add randomness
    dr = -1
    if player == 2:
        dr = 1

    for i in range(positions.shape[0]):
        position = positions[i]

        row = position // 8
        col = position % 8
        ran = list(range(-1, 2))
        random.shuffle(ran)
        for dc in ran:
            new_row, new_col = row + dr, col + dc
            if not (
                (0 <= new_row) & (new_row < 8) & (0 <= new_col) & (new_col < 8)
            ):  # if the new position is not in bounds, skip to the next direction
                continue
            new_position = new_row * 8 + new_col
            if dc == 0:  # moving straight
                if board[new_position] == 0:
                    return (position, new_position)
            else:  # capturing
                if board[new_position] != player:
                    return (position, new_position)

    # if no legal moves are found after iterating all positions and directions, return None
    return (0, 0)


@cython.ccall
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.locals(
    player=cython.int,
    dr=cython.int,
    dc=cython.int,
    new_row=cython.int,
    new_col=cython.int,
    position=cython.int,
    row=cython.int,
    col=cython.int,
    board=cython.int[:],
)
def get_legal_actions(player, board) -> cython.list:
    """
    Get all legal actions for the current player.

    :return: A list of legal actions as tuples (from_position, to_position).
    In case of a terminal state, an empty list is returned.
    """
    legal_actions: cython.list = []
    dr = -1
    if player == 2:
        dr = 1

    for position in range(board.shape[0]):
        if board[position] == player:
            row = position // 8
            col = position % 8
            for dc in range(-1, 2):
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:  # Check if the new position is within the board
                    new_position = new_row * 8 + new_col

                    if dc == 0:  # moving straight
                        if board[new_position] == 0:
                            legal_actions.append((position, new_position))

                    else:  # diagonal capture / move
                        if board[new_position] != player:
                            legal_actions.append((position, new_position))
    return legal_actions


@cython.ccall
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def evaluate_move(
    board: cython.int[:], from_position: cython.int, to_position: cython.int, player: cython.int
) -> cython.int:
    score: cython.int = 0

    # Use lorentz_values for base_value, the max value is 36, so we add 36 to the score to make sure it is non-negative
    if player == 2:
        score = 36 + (lorentz_values[to_position] - lorentz_values[from_position])
    else:
        # Player 1 views the lorenz_values in reverse
        score = 36 + (lorentz_values[63 - to_position] - lorentz_values[63 - from_position])

    # Reward capturing
    if board[to_position] == 3 - player:
        score = int(score**2)  # square score if it's a capture
        # An antidecisive move
        if (player == 1 and from_position > 56) or (player == 2 and from_position < 8):
            score += 1000000  # Add a very high score if the move is antidecisive

    # Reward safe positions
    if is_safe(to_position, player, board):
        score *= 2  # Add base_value again if the position is safe

    # Reward decisive moves
    # The decisive condition checks for a piece on the penultimate row that can move to the opponent's final row without the opportunity of being captured.
    if (player == 1 and (8 <= from_position <= 16)) or (player == 2 and (48 <= from_position <= 56)):
        score += 1000000  # Add a very high score if the move is decisive

    return score


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
    a: cython.int = 100,
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
    pieces: cython.long[:] = np.where(state.board > 0)[0]  # get all pieces from the board
    new_position: cython.int
    new_col: cython.int
    new_row: cython.int
    y: cython.int
    x: cython.int
    piece: cython.int
    dr: cython.int
    dc: cython.int

    for i in range(pieces.shape[0]):
        piece = board[pieces[i]]
        x = pieces[i] // 8
        y = pieces[i] % 8

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
    m_lorenz: cython.double = 1.0,
    m_mobility: cython.double = 1.0,
    m_safe: cython.double = 1.0,
    m_endgame: cython.double = 1.0,
    m_cap: cython.double = 2.0,
    m_piece_diff: cython.double = 1.0,
    m_opp_disc: cython.double = 0.9,
    m_decisive: cython.double = 100.0,
    m_antidecisive: cython.double = 100.0,
    a: cython.int = 200,
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
    antidecisive_values: cython.double = 0
    caps: cython.double = 0
    opponent: cython.int = 3 - player
    board: cython.int[:] = state.board
    positions: cython.long[:] = np.where(state.board > 0)[0]  # get all pieces from the board
    multiplier: cython.double
    piece_value: cython.double
    mob_val: cython.double
    position: cython.int
    piece: cython.int

    for i in range(positions.shape[0]):
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

    if m_decisive != 0.0 or m_antidecisive != 0.0:
        player_1_decisive, player_2_decisive, player_1_antidecisive, player_2_antidecisive = is_decisive(
            state
        )

        if player == 1:
            decisive_values = m_decisive if player_1_decisive else 0
            decisive_values -= m_decisive if player_2_decisive else 0
            antidecisive_values = m_antidecisive if player_1_antidecisive else 0
            antidecisive_values -= m_antidecisive if player_2_antidecisive else 0
        else:  # self.player == 2
            decisive_values = m_decisive if player_2_decisive else 0
            decisive_values -= m_decisive if player_1_decisive else 0
            antidecisive_values = m_antidecisive if player_2_antidecisive else 0
            antidecisive_values -= m_antidecisive if player_1_antidecisive else 0

    if m_cap >= 0:
        caps = count_capture_moves(board, np.where(state.board == player)[0], player) - count_capture_moves(
            board, np.where(state.board == opponent)[0], opponent
        )

    eval_value: cython.double = (
        decisive_values
        + antidecisive_values
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
@cython.infer_types(True)
@cython.locals(
    state=cython.object,
    player_1_decisive=cython.bint,
    player_2_decisive=cython.bint,
    player_1_antidecisive=cython.bint,
    player_2_antidecisive=cython.bint,
    pos=cython.int,
    i=cython.int,
    board=cnp.ndarray,
    player_positions=cython.long[:],  # to specify a 1-D memoryview type, can be more efficient
)
def is_decisive(
    state: cython.object,
) -> tuple[cython.int, cython.int, cython.int, cython.int]:
    board = state.board
    player_1_decisive = player_2_decisive = player_1_antidecisive = player_2_antidecisive = 0

    # Check for player 1
    player_positions = np.where(board[bef_last_rows_1] == 1)[0] + bef_last_rows_1[0]

    for i in range(player_positions.shape[0]):
        pos = player_positions[i]
        if state.player == 1:
            player_1_decisive = 1
            break
        else:
            if (pos % 8 > 0 and board[pos - 7] == 2) or (pos % 8 < 7 and board[pos - 9] == 2):
                player_2_antidecisive = 1
                break
            else:
                player_1_decisive = 1
                break

    # Check for player 2
    player_positions = np.where(board[bef_last_rows_2] == 2)[0] + bef_last_rows_2[0]
    for i in range(player_positions.shape[0]):
        pos = player_positions[i]
        if state.player == 2:
            player_2_decisive = 1
            break
        else:
            if (pos % 8 > 0 and board[pos + 7] == 1) or (pos % 8 < 7 and board[pos + 9] == 1):
                player_1_antidecisive = 1
                break
            else:
                player_2_decisive = 1
                break

    return player_1_decisive, player_2_decisive, player_1_antidecisive, player_2_antidecisive


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
    positions=cython.long[:],
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
    n: cython.int = positions.shape[0]

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

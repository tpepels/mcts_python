from typing import Tuple
import numpy as np
from games.gamestate import GameState, normalize, win, loss, draw
import random


class BreakthroughGameState(GameState):
    players_bitstrings = [random.randint(1, 2**64 - 1) for _ in range(3)]  # 0 is for the empty player
    zobrist_table = [[[random.randint(1, 2**64 - 1) for _ in range(3)] for _ in range(8)] for _ in range(8)]

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
        board = np.zeros(64, dtype=np.int8)
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

        positions = np.where(self.board == self.player)[0]
        random.shuffle(positions)  # shuffle the positions to add randomness
        dr = -1 if self.player == 1 else 1
        directions = [-1, 0, 1]
        random.shuffle(directions)  # shuffle the directions to add randomness

        for position in positions:
            row, col = divmod(position, 8)

            for dc in directions:
                new_row, new_col = row + dr, col + dc
                in_bounds = (0 <= new_row) & (new_row < 8) & (0 <= new_col) & (new_col < 8)

                if not in_bounds:  # if the new position is not in bounds, skip to the next direction
                    continue

                new_position = new_row * 8 + new_col

                if dc == 0:  # moving straight
                    if self.board[new_position] == 0:
                        return (position, new_position)
                else:  # capturing
                    if self.board[new_position] != self.player:
                        return (position, new_position)

        # if no legal moves are found after iterating all positions and directions, return None
        return None

    def yield_legal_actions(self):
        """
        Yield all legal actions for the current player.

        :yield: Legal actions as tuples (from_position, to_position). In case of a terminal state, an empty sequence is returned.
        """
        positions = np.where(self.board == self.player)[0]
        random.shuffle(positions)  # Shuffle positions

        dr = -1 if self.player == 1 else 1

        for position in positions:
            row, col = divmod(position, 8)
            dc_values = [-1, 0, 1]
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

        legal_actions = []
        positions = np.where(self.board == self.player)[0]
        row, col = divmod(positions, 8)
        dr = -1 if self.player == 1 else 1

        for dc in [-1, 0, 1]:
            new_row, new_col = row + dr, col + dc
            in_bounds = (0 <= new_row) & (new_row < 8) & (0 <= new_col) & (new_col < 8)

            valid_positions = positions[in_bounds]  # update the positions based on in_bound mask
            new_positions = new_row * 8 + new_col
            valid_new_positions = new_positions[in_bounds]  # update the new_positions based on in_bound mask

            if len(valid_new_positions) == 0:  # if there are no valid new positions, continue to next dc
                continue

            if dc == 0:  # moving straight
                mask = self.board[valid_new_positions] == 0
            else:  # capturing
                mask = self.board[valid_new_positions] != self.player

            legal_actions.extend(zip(valid_positions[mask], valid_new_positions[mask]))

        return legal_actions

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

    def evaluate_move(self, move):
        """
        Evaluate the given move using a simple heuristic: each step forward is worth 1 point,
        and capturing an opponent's piece is worth 2 points.

        :param move: The move to evaluate.
        :return: The heuristic value of the move.
        """
        from_position, to_position = move

        # Player 1 views the lorenz_values in reverse
        if self.player == 1:
            to_position = 63 - to_position
            from_position = 63 - from_position

        # Use lorentz_values for base_value
        score = lorentz_values[to_position] - lorentz_values[from_position]

        # Reward safe positions
        if is_safe(move[1], self.player, self.board):
            score += score  # Add base_value again if the position is safe

        # Reward capturing
        if self.is_capture(move):
            score ^= 2  # square score if it's a capture

        return score

    def visualize(self):
        """
        Visualize the board for the Breakthrough game.

        :param state: The Breakthrough game state.
        :param characters: If True, print the board with characters, otherwise print raw values.
        """
        result = ""
        cell_representation = {0: ".", 1: "W", 2: "B"}
        column_letters = "  " + " ".join("ABCDEFGH") + "\n"  # Use chess-style notation

        for i in range(8):
            row = [cell_representation.get(piece, ".") for piece in self.board[i * 8 : i * 8 + 8]]
            formatted_row = " ".join(row)
            result += f"{i + 1} {formatted_row}\n"

        return column_letters + result + "hash: " + str(self.board_hash)

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
        return 2**18

    def __repr__(self) -> str:
        return "breakthrough"


def to_chess_notation(index):
    """Transform a board index into chess notation."""
    row, col = divmod(index, 8)
    return f"{chr(col + 65)}{row + 1}"


def evaluate_breakthrough(
    state: BreakthroughGameState,
    player: int,
    m_piece: float = 0.25,
    m_dist: float = 0.25,
    m_near: float = 0.25,
    m_blocked: float = 0.25,
    m_opp_disc: float = 0.9,
    a: int = 100,
    norm: bool = False,
):
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

    opponent = 3 - player

    metrics = {
        player: {"pieces": 0, "distance": 0, "near_opponent_side": 0, "blocked": 0},
        opponent: {"pieces": 0, "distance": 0, "near_opponent_side": 0, "blocked": 0},
    }

    pieces = np.where(state.board > 0)[0]  # get all pieces from the board
    for position in pieces:
        piece = state.board[position]

        x, y = divmod(position, 8)

        metrics[piece]["pieces"] += 1
        dist = 7 - x if piece == 2 else x
        metrics[piece]["distance"] += dist

        if (piece == 2 and x >= 4) or (piece == 1 and x <= 3):
            metrics[piece]["near_opponent_side"] = min(
                dist, metrics[piece]["near_opponent_side"]
            )  # the closer the better

        dr = -1 if piece == 1 else 1  # Determine the direction of movement based on the current player
        blocked = True
        for dc in (-1, 0, 1):
            new_row, new_col = x + dr, y + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                new_position = new_row * 8 + new_col
                if state.board[new_position] == 0 or (dc != 0 and state.board[new_position] != piece):
                    blocked = False
                    break

        metrics[piece]["blocked"] += 1 if blocked else 0

    piece_difference = (metrics[player]["pieces"] - metrics[opponent]["pieces"]) / (
        metrics[player]["pieces"] + metrics[opponent]["pieces"]
    )
    avg_distance_difference = (metrics[opponent]["distance"] - metrics[player]["distance"]) / (
        metrics[player]["pieces"] + metrics[opponent]["pieces"]
    )
    near_opponent_side_difference = (
        metrics[player]["near_opponent_side"] - metrics[opponent]["near_opponent_side"]
    ) / (metrics[player]["pieces"] + metrics[opponent]["pieces"])

    blocked_difference = (metrics[opponent]["blocked"] - metrics[player]["blocked"]) / (
        metrics[player]["pieces"] + metrics[opponent]["pieces"]
    )

    eval_value = (
        m_piece * piece_difference
        + m_dist * avg_distance_difference
        + m_near * near_opponent_side_difference
        + m_blocked * blocked_difference
    ) * (m_opp_disc if state.player == opponent else 1)

    if norm:
        return normalize(eval_value, a)
    else:
        return eval_value


def evaluate_breakthrough_lorenz(
    state: BreakthroughGameState,
    player: int,
    m_lorenz: float = 1.0,
    m_mobility: float = 1.0,
    m_blocked: float = 1.0,
    m_safe: float = 1.0,
    m_endgame: float = 1.0,
    m_cap: float = 2.0,
    m_cap_move: float = 2.0,
    m_opp_disc: float = 0.9,
    a: int = 200,
    norm: bool = True,
):
    """
    Evaluates the current game state using an enhanced Lorenz evaluation function,
    which takes into account the positioning of pieces, their mobility, the safety of their positions,
    whether the piece is blocked, the phase of the endgame, and the potential to capture opposing pieces.

    :param state: The game state to evaluate.
    :param player: The player to evaluate for (1 or 2).
    :param m_lorenz: Weight assigned to the Lorenz value of the board configuration.
    :param m_mobility: Weight assigned to the mobility of the player's pieces.
    :param m_blocked: Weight assigned to the blocked pieces of the player.
    :param m_safe: Weight assigned to the safety of the player's pieces.
    :param m_endgame: Weight assigned to the endgame phase.
    :param m_cap: Weight assigned to the capture difference between the player and the opponent.
    :param m_cap_move: Weight assigned to the difference in potential capture moves between the player and the opponent.
    :param m_opp_disc: Multiplier for the evaluation score if it's the opponent's turn.
    :param a: Normalization factor for the evaluation score.
    :param norm: If True, the function will return a normalized evaluation score.

    :return: The evaluation score for the given player.
    """
    board_values = 0
    mobility_values = 0
    blocked_values = 0
    safety_values = 0
    endgame_values = 0
    piece_diff = 0
    pieces = 0
    opponent = 3 - player

    for position, piece in enumerate(state.board):
        if piece == 0:
            continue

        # Player or opponent
        multiplier = 1 if piece == player else -1
        pieces += 1
        piece_diff += multiplier

        piece_value = lorentz_values[position] if piece == 2 else lorentz_values[63 - position]
        board_values += multiplier * piece_value

        if m_mobility != 0 or m_blocked != 0:
            mob_val = piece_mobility(position, piece, state.board)
            mobility_values += multiplier * mob_val
            if mob_val == 0:  # Keep track of the number of blocked pieces
                blocked_values -= multiplier

        if m_safe != 0 and is_safe(position, piece, state.board):
            safety_values += multiplier * m_safe * piece_value

    if m_endgame != 0 and pieces < 12:
        endgame_values = m_endgame * piece_diff

    my_caps, my_cap_moves = count_capture_moves(state, player) if m_cap != 0 or m_cap_move != 0 else (0, 0)
    opp_caps, opp_cap_moves = (
        count_capture_moves(state, opponent) if m_cap != 0 or m_cap_move != 0 else (0, 0)
    )

    eval_value = (
        endgame_values
        + m_lorenz * board_values
        + m_mobility * mobility_values
        + m_blocked * blocked_values
        + safety_values
        + m_cap * (my_caps - opp_caps)
        + m_cap_move * (my_cap_moves - opp_cap_moves)
    ) * (m_opp_disc if state.player == opponent else 1)

    if norm:
        return normalize(eval_value, a)
    else:
        return eval_value


BL_DIR = ((1, 0), (1, -1), (1, 1))
WH_DIR = ((-1, 0), (-1, -1), (-1, 1))


def piece_mobility(position, player, board):
    """
    Calculates the mobility of a piece at the given position.

    :param position: The position of the piece on the board.
    :param player: The player to which the piece belongs (1 or 2).
    :param board: The game state.
    :return: The mobility value of the piece.
    """
    row, col = divmod(position, 8)
    mobility = 0

    # The directions of the moves depend on the player.
    # White (1) moves up (to lower rows), Black (2) moves down (to higher rows).
    if player == 1:  # White
        directions = WH_DIR  # Forward, and diagonally left and right
    else:  # Black
        directions = BL_DIR  # Forward, and diagonally left and right

    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < 8 and 0 <= new_col < 8:
            new_position = new_row * 8 + new_col
            if (dc == 0 and board[new_position] == 0) or (
                abs(dc) == 1 and (board[new_position] == 0 or board[new_position] != player)
            ):
                mobility += 1

    return mobility


def is_safe(position, player, board):
    """
    Determines if a piece at a given position is safe from being captured.

    :param position: The position of the piece.
    :param player: The player to which the piece belongs (1 or 2).
    :param state: The game state.
    :return: True if the piece is safe, False otherwise.
    """
    x, y = divmod(position, 8)
    row_offsets = [-1, 1] if player == 1 else [1, -1]
    col_offsets = [-1, 1]
    attackers = 0
    defenders = 0

    opponent = 1 if player == 2 else 2

    for i in range(2):
        new_x = x + row_offsets[0]
        new_y = y + col_offsets[i]

        if 0 <= new_x < 8 and 0 <= new_y < 8:
            new_position = new_x * 8 + new_y
            piece = board[new_position]

            if piece == opponent:
                attackers += 1

        new_x = x + row_offsets[1]
        new_y = y + col_offsets[i]

        if 0 <= new_x < 8 and 0 <= new_y < 8:
            new_position = new_x * 8 + new_y
            piece = board[new_position]

            if piece == player:
                defenders += 1

    return attackers <= defenders


def count_capture_moves(game_state: BreakthroughGameState, player: int) -> Tuple[int, int]:
    """Count the number of pieces that can capture and the total number of possible capture moves for a given player.

    :param game_state: The game state instance.
    :param player: The player (1 or 2) to count capture moves for.
    :return: The number of pieces that can capture and the total number of possible capture moves for the player.
    """
    num_pieces_that_can_capture = 0
    total_capture_moves = 0
    positions = np.where(game_state.board == player)[0]

    for position in positions:
        row, col = divmod(position, 8)
        dr = -1 if player == 1 else 1

        piece_capture_moves = 0
        for dc in [-1, 1]:  # Only diagonal movements can result in capture
            new_row, new_col = row + dr, col + dc
            if (
                0 <= new_row < 8 and 0 <= new_col < 8
            ):  # Check if the new position is within the board boundaries
                new_position = new_row * 8 + new_col
                # Check if the new position contains an opponent's piece
                if game_state.board[new_position] == 3 - player:
                    piece_capture_moves += 1
        total_capture_moves += piece_capture_moves
        if piece_capture_moves > 0:
            num_pieces_that_can_capture += 1

    return num_pieces_that_can_capture, total_capture_moves


# List of values representing the importance of each square on the board. In view of player 2.
lorentz_values = np.array(
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
    dtype=int,
)
# Normalize the lorenz values so it requires less tuning to combine with other heuristics
lorentz_values = (lorentz_values - np.min(lorentz_values)) / (np.max(lorentz_values) - np.min(lorentz_values))

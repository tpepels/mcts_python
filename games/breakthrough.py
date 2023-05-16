from pprint import pprint
from games.gamestate import GameState, normalize


def visualize_breakthrough(state, characters=True):
    """
    Visualize the board for the Breakthrough game.

    :param state: The Breakthrough game state.
    :param characters: If True, print the board with characters, otherwise print raw values.
    """
    result = ""
    if characters:
        cell_representation = {0: ".", 1: "W", 2: "B"}
    else:
        cell_representation = {0: "0", 1: "100", 2: "200"}
        result += "[\n"

    for i in range(8):
        row = [cell_representation.get(piece // 100, ".") for piece in state.board[i * 8 : i * 8 + 8]]
        formatted_row = " ".join(row) if characters else ", ".join(row)
        result += f"{i} {formatted_row}\n"

    if not characters:
        result += "]"

    column_numbers = "  " + " ".join(map(str, range(0, 8))) + "\n"

    return column_numbers + result


class BreakthroughGameState(GameState):
    """
    This class represents the game state for the Breakthrough board game.
    Breakthrough is a two-player game played on an 8x8 board.
    Each player starts with 16 pieces occupying the first two rows.
    The goal is to move one of your pieces to the opponent's home row.
    """

    def __init__(self, board=None, player=1):
        """
        Initialize the game state with the given board configuration.
        If no board is provided, the game starts with the default setup.

        :param board: An optional board configuration.
        :param player: The player whose turn it is (1 or 2).
        """
        if board is None:
            self.board = [0] * 64
            for i in range(16):
                self.board[i] = 200 + i  # Initialize player 2 pieces
                self.board[48 + i] = 100 + i  # Initialize player 1 pieces
        else:
            self.board = board

        self.player = player

    def apply_action(self, action):
        """
        Apply the given action to create a new game state. The current state is not altered by this method.
        Actions are represented as a tuple (from_position, to_position).

        :param action: The action to apply.
        :return: A new game state with the action applied.
        """
        from_position, to_position = action
        new_board = self.board.copy()

        piece = new_board[from_position]
        new_board[from_position] = 0  # Remove the piece from its current position
        new_board[to_position] = piece  # Place the piece at its new position

        if not self._is_terminal(new_board):
            return BreakthroughGameState(
                new_board, 3 - self.player
            )  # Return new state with the other player's turn
        else:
            return BreakthroughGameState(new_board, self.player)  # Don't change the turn, board is terminal

    def get_legal_actions(self):
        """
        Get all legal actions for the current player.

        :return: A list of legal actions as tuples (from_position, to_position). In case of a terminal state, an empty list is returned.
        """

        if self.is_terminal():
            return []

        legal_actions = []

        for position, piece in enumerate(self.board):
            if piece == 0 or piece // 100 != self.player:
                continue

            row, col = divmod(position, 8)
            dr = (
                -1 if self.player == 1 else 1
            )  # Determine the direction of movement based on the current player

            for dc in (-1, 0, 1):  # Check all possible moves
                new_row, new_col = row + dr, col + dc

                if not in_bounds(new_row, new_col):  # Skip if the move is out of bounds
                    continue

                new_position = new_row * 8 + new_col
                if (dc == 0 and self.board[new_position] == 0) or (  # Move forward to an empty cell
                    dc != 0 and self.board[new_position] // 100 != self.player  # Capture
                ):
                    legal_actions.append((position, new_position))

        return legal_actions

    def is_terminal(self):
        """
        Check if the current game state is terminal (i.e., a player has won).

        :return: True if the game state is terminal, False otherwise.
        """
        return self._is_terminal(self.board)

    def _is_terminal(self, board):
        """
        Check if a game state is terminal (i.e., a player has won).

        :return: True if the game state is terminal, False otherwise.
        """
        # Check for player 2 winning
        for piece in board[0:8]:
            if piece // 100 == 1:  # Player 2 pieces have values 200 and above
                return True

        # Check for player 1 winning
        for piece in board[56:64]:
            if piece // 100 == 2:  # Player 1 pieces have values 100 and above
                return True

        return False

    def get_reward(self):
        """
        Get the reward for the current game state.
        The reward is 1 for player 1 if they have won, -1 for player 2 if they have won, and 0 otherwise.

        :return: The reward for the current game state.
        """
        # Check for player 1 winning
        for piece in self.board[0:8]:
            if piece // 100 == 1:  # Player 2 pieces have values 200 and above
                return 1

        # Check for player 2 winning
        for piece in self.board[56:64]:
            if piece // 100 == 2:  # Player 1 pieces have values 100 and above
                return -1

        return 0

    def is_capture(self, move):
        """
        Check if the given move results in a capture.

        :param move: The move to check as a tuple (from_position, to_position).
        :return: True if the move results in a capture, False otherwise.
        """
        from_position, to_position = move

        # Check if the destination cell contains an opponent's piece
        if self.board[to_position] // 100 == 3 - self.player:
            from_row, from_col = divmod(from_position, 8)
            to_row, to_col = divmod(to_position, 8)

            # Ensure the move is diagonal
            if abs(from_row - to_row) == 1 and abs(from_col - to_col) == 1:
                return True

        return False


# The evaluate function takes a Breakthrough game state and the player number (1 or 2) as input.
# It computes the number of pieces for each player, the average distance of the pieces to the opponent's side,
# and the number of pieces that are close to the opponent's side.
# It then calculates the difference between the players for each metric and combines them
# with weights to produce an evaluation score.


def evaluate(state, player, m=(0.4, 0.4, 0.2, 0.1), a=100, norm=False):
    opponent = 3 - player

    metrics = {
        player: {"pieces": 0, "distance": 0, "near_opponent_side": 0, "blocked": 0},
        opponent: {"pieces": 0, "distance": 0, "near_opponent_side": 0, "blocked": 0},
    }

    for position in range(64):
        piece = state.board[position]
        x, y = divmod(position, 8)
        piece_player = piece // 100

        if piece_player in [player, opponent]:
            metrics[piece_player]["pieces"] += 1
            dist = 7 - x if piece_player == 2 else x
            metrics[piece_player]["distance"] += dist
            if (piece_player == 2 and x >= 5) or (piece_player == 1 and x <= 2):
                metrics[piece_player]["near_opponent_side"] += 7 - dist  # the close the better
            if is_piece_blocked(position, state.board):
                metrics[piece_player]["blocked"] += 1

    if metrics[player]["pieces"] == 0 or metrics[opponent]["pieces"] == 0:
        return 0

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

    pprint(metrics)

    eval_value = (
        m[0] * piece_difference
        + m[1] * avg_distance_difference
        + m[2] * near_opponent_side_difference
        + m[3] * blocked_difference
    )

    if norm:
        return normalize(eval_value, a)
    else:
        return eval_value


# Lorenz evaluation functions

# List of values representing the importance of each square on the board.
lorentz_values = [
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
]


def lorenz_evaluation(state, player, m=(0.5,), a=200, norm=True):
    """
    Evaluates the current game state using the Lorenz evaluation function, which
    takes into account the positioning of pieces on the board.

    :param state: The game state to evaluate.
    :param player: The player to evaluate for (1 or 2).
    :return: The evaluation score for the given player.
    """

    p1eval = 0
    p2eval = 0

    for position in range(64):
        piece = state.board[position]

        if piece // 100 == 2:
            base_value = lorentz_values[position]
            p2eval += base_value

            if is_safe(position, 1, state.board):
                p2eval += m[0] * base_value

        elif piece // 100 == 1:
            base_value = lorentz_values[63 - position]  # Mirror the position for the opponent
            p1eval += base_value

            if is_safe(position, 2, state.board):
                p1eval += m[0] * base_value

    if player == 1:
        eval_value = p1eval - p2eval
    else:
        eval_value = p2eval - p1eval

    if norm:
        return normalize(eval_value, a)
    else:
        return eval_value


def lorenz_enhanced_evaluation(state, player, m=(0.5, 10, 0.5, 0.5), a=200, norm=True):
    """
    Evaluates the current game state using an enhanced Lorenz evaluation function,
    which takes into account the positioning of pieces, their mobility, and the
    endgame phase.

    :param state: The game state to evaluate.
    :param player: The player to evaluate for (1 or 2).
    :return: The evaluation score for the given player.
    """
    board_values = 0
    mobility_values = 0
    safety_values = 0
    endgame_values = 0

    opponent = 3 - player

    for position, piece in enumerate(state.board):
        if piece == 0:
            continue

        piece_player = piece // 100
        piece_value = lorentz_values[position] if piece_player == 2 else lorentz_values[63 - position]

        if piece_player == player:
            board_values += piece_value
            mobility_values += piece_mobility(position, piece_player, state.board)
        else:
            board_values -= piece_value
            mobility_values -= piece_mobility(position, piece_player, state.board)

        if is_safe(position, piece_player, state.board):
            safety_bonus = m[0] * piece_value
            safety_values += safety_bonus if piece_player == player else -safety_bonus

    if is_endgame(state.board):
        endgame_values = m[1] * (
            sum(1 for piece in state.board if piece != 0 and piece // 100 == player)
            - sum(1 for piece in state.board if piece != 0 and piece // 100 == opponent)
        )

    eval_value = endgame_values + m[2] * board_values + m[3] * mobility_values + safety_values

    if norm:
        return normalize(eval_value, a)
    else:
        return eval_value


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
    # White (100-199) moves up (to lower rows), Black (200+) moves down (to higher rows).
    if player == 1:  # White
        directions = ((-1, 0), (-1, -1), (-1, 1))  # Forward, and diagonally left and right
    else:  # Black
        directions = ((1, 0), (1, -1), (1, 1))  # Forward, and diagonally left and right

    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if in_bounds(new_row, new_col):
            new_position = new_row * 8 + new_col
            if board[new_position] // 100 != player:
                if (
                    board[new_position] == 0 or abs(dc) == 1
                ):  # Move forward to an empty cell or capture diagonally
                    mobility += 1

    return mobility


def is_endgame(board):
    """
    Determines if the game is in the endgame phase.

    :param state: The game state.
    :return: True if the game is in the endgame phase, False otherwise.
    """
    pieces_count = sum(1 for cell in board if cell != 0)
    return pieces_count <= 16  # i.e. half of the pieces are remaining on the board


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

    opponent = 100 if player == 2 else 200

    for i in range(2):
        new_x = x + row_offsets[0]
        new_y = y + col_offsets[i]

        if in_bounds(new_x, new_y):
            new_position = new_x * 8 + new_y
            piece = board[new_position] // 100

            if piece == (opponent // 100):
                attackers += 1

        new_x = x + row_offsets[1]
        new_y = y + col_offsets[i]

        if in_bounds(new_x, new_y):
            new_position = new_x * 8 + new_y
            piece = board[new_position] // 100

            if piece == player:
                defenders += 1

    return attackers <= defenders


def in_bounds(x, y):
    """
    Determines if the given coordinates (x, y) are within the board bounds.

    :param x: The x-coordinate.
    :param y: The y-coordinate.
    :return: True if the coordinates are within the board bounds, False otherwise.
    """
    return 0 <= x < 8 and 0 <= y < 8


def get_player_pieces(board, player):
    """
    Get the positions of all pieces belonging to a player.

    :param state: The Breakthrough game state.
    :param player: The player number (1 for white, 2 for black).
    :return: A list of tuples, where each tuple represents the row and column of a piece.
    """
    piece_positions = []
    for pos in range(64):
        if board[pos] // 100 == player:
            piece_positions.append(pos)
    return piece_positions


def is_piece_blocked(position, board):
    """
    Check if the given piece is blocked and cannot move one position forward.

    :param position: The position of the piece to check.
    :return: True if the piece is blocked, False otherwise.
    """
    piece = board[position]
    if piece == 0:
        raise ValueError("Invalid piece")

    player = piece // 100
    row, col = divmod(position, 8)
    dr = -1 if player == 1 else 1  # Determine the direction of movement based on the current player

    for dc in [-1, 0, 1]:
        new_row, new_col = row + dr, col + dc
        if in_bounds(new_row, new_col):
            new_position = new_row * 8 + new_col
            if board[new_position] == 0 or (dc != 0 and board[new_position] // 100 != player):
                return False

    return True  # The piece is blocked

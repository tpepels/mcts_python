from games.gamestate import GameState


def visualize_breakthrough(state):
    board = state.board
    for i in range(8):
        row = []
        for j in range(8):
            piece = board[i][j]
            if piece // 100 == 1:
                row.append("P")
            elif piece // 100 == 2:
                row.append("Q")
            else:
                row.append(".")
        print(" ".join(row))
    print("\n")


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
        Apply the given action to the game state and return a new game state.
        Actions are represented as a tuple (from_position, to_position).

        :param action: The action to apply.
        :return: A new game state with the action applied.
        """
        from_position, to_position = action
        piece = self.board[from_position]
        self.board[from_position] = 0  # Remove the piece from its current position
        self.board[to_position] = piece  # Place the piece at its new position
        return BreakthroughGameState(
            self.board, 3 - self.player
        )  # Return new state with the other player's turn

    def get_legal_actions(self):
        """
        Get all legal actions for the current player.

        :return: A list of legal actions as tuples (from_position, to_position).
        """
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
                if not self.in_bounds(new_row, new_col):  # Skip if the move is out of bounds
                    continue

                new_position = new_row * 8 + new_col
                # If the destination position is empty or occupied by an opponent's piece, the move is legal
                if self.board[new_position] == 0 or self.board[new_position] // 100 != self.player:
                    legal_actions.append((position, new_position))

        return legal_actions

    def is_terminal(self):
        """
        Check if the current game state is terminal (i.e., one player has won).

        :return: True if the game state is terminal, False otherwise.
        """
        return any(
            (piece // 100 == self.player and (piece % 100) // 8 == (7 if self.player == 1 else 0))
            for piece in self.board
        )

    def get_reward(self):
        """
        Get the reward for the current game state.
        The reward is 1 for player 1 if they have won, -1 for player 2 if they have won, and 0 otherwise.

        :return: The reward for the current game state.
        """
        if self.is_terminal():
            return 1 if self.player == 1 else -1
        return 0

    def in_bounds(self, r, c):
        """
        Check if a given position is within the bounds of the board.

        :param r: The row index.
        :param c: The column index.
        :return: True if the position is in bounds, False otherwise.
        """
        return 0 <= r < 8 and 0 <= c < 8


# The evaluate function takes a Breakthrough game state and the player number (1 or 2) as input.
# It computes the number of pieces for each player, the average distance of the pieces to the opponent's side,
# and the number of pieces that are close to the opponent's side.
# It then calculates the difference between the players for each metric and combines them
# with weights to produce an evaluation score.


def evaluate(state, player):
    opponent = 3 - player
    player_pieces = 0
    opponent_pieces = 0
    player_distance = 0
    opponent_distance = 0
    player_near_opponent_side = 0
    opponent_near_opponent_side = 0

    for x in range(8):
        for y in range(8):
            piece = state.board[x][y]

            if piece == player:
                player_pieces += 1
                player_distance += 7 - x if player == 1 else x

                if player == 1 and x >= 6:
                    player_near_opponent_side += 1
                elif player == 2 and x <= 1:
                    player_near_opponent_side += 1

            elif piece == opponent:
                opponent_pieces += 1
                opponent_distance += 7 - x if opponent == 1 else x

                if opponent == 1 and x >= 6:
                    opponent_near_opponent_side += 1
                elif opponent == 2 and x <= 1:
                    opponent_near_opponent_side += 1

    if player_pieces == 0 or opponent_pieces == 0:
        return 0

    piece_difference = (player_pieces - opponent_pieces) / (player_pieces + opponent_pieces)
    avg_distance_difference = (opponent_distance - player_distance) / (player_pieces + opponent_pieces)
    near_opponent_side_difference = (player_near_opponent_side - opponent_near_opponent_side) / (
        player_pieces + opponent_pieces
    )

    return 0.4 * piece_difference + 0.4 * avg_distance_difference + 0.2 * near_opponent_side_difference


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


def lorenz_evaluation(state, player):
    """
    Evaluates the current game state using the Lorenz evaluation function, which
    takes into account the positioning of pieces on the board.

    :param state: The game state to evaluate.
    :param player: The player to evaluate for (1 or 2).
    :return: The evaluation score for the given player.
    """
    opponent = 3 - player

    p1eval = 0
    p2eval = 0

    for x in range(8):
        for y in range(8):
            piece = state.board[x][y]

            if piece == player:
                base_value = lorentz_values[x * 8 + y]
                p1eval += base_value

                if is_safe(x, y, player, state):
                    p1eval += 0.5 * base_value

            elif piece == opponent:
                base_value = lorentz_values[(7 - x) * 8 + y]
                p2eval += base_value

                if is_safe(x, y, opponent, state):
                    p2eval += 0.5 * base_value

    if player == 1:
        eval_value = p1eval - p2eval
    else:
        eval_value = p2eval - p1eval

    return eval_value


def piece_mobility(position, player, state):
    """
    Calculates the mobility of a piece at the given position.

    :param position: The position of the piece on the board.
    :param player: The player to which the piece belongs (1 or 2).
    :param state: The game state.
    :return: The mobility value of the piece.
    """
    row, col = divmod(position, 8)
    mobility = 0

    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        new_row, new_col = row + dr, col + dc
        if in_bounds(new_row, new_col):
            new_position = new_row * 8 + new_col
            if state[new_position] == 0 or state[new_position] // 100 != player:
                mobility += 1

    return mobility


def is_endgame(state):
    """
    Determines if the game is in the endgame phase.

    :param state: The game state.
    :return: True if the game is in the endgame phase, False otherwise.
    """
    pieces_count = sum(1 for cell in state if cell != 0)
    return pieces_count <= 8  # i.e. half of the pieces are remaining on the board


def lorenz_enhanced_evaluation(state, player):
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
    for position, piece in enumerate(state):
        if piece == 0:
            continue

        piece_player = piece // 100
        piece_value = lorentz_values[position] if piece_player == 2 else lorentz_values[63 - position]

        if piece_player == player:
            board_values += piece_value
            mobility_values += piece_mobility(position, player, state)
        else:
            board_values -= piece_value

        if is_safe(position, position, piece_player, state):
            safety_bonus = 0.5 * piece_value
            board_values += safety_bonus if piece_player == player else -safety_bonus

    if is_endgame(state):
        endgame_bonus = sum(1 for piece in state if piece != 0 and piece // 100 == player) * 10
        board_values += endgame_bonus

    return board_values + mobility_values


def is_safe(x, y, player, state):
    """
    Determines if a piece at position (x, y) is safe from being captured.

    :param x: The x-coordinate of the piece.
    :param y: The y-coordinate of the piece.
    :param player: The player to which the piece belongs (1 or 2).
    :param state: The game state.
    :return: True if the piece is safe, False otherwise.
    """
    row_offsets = [-1, -1, 1, 1]
    col_offsets = [-1, 1, -1, 1]
    attackers = 0
    defenders = 0

    opponent = 3 - player

    for i in range(4):
        new_x = x + row_offsets[i]
        new_y = y + col_offsets[i]

        if in_bounds(new_x, new_y):
            piece = state.board[new_x][new_y]

            if player == 1:
                if i < 2 and piece == opponent:
                    attackers += 1
                elif i >= 2 and piece == player:
                    defenders += 1
            else:
                if i < 2 and piece == player:
                    defenders += 1
                elif i >= 2 and piece == opponent:
                    attackers += 1

    return attackers <= defenders


def in_bounds(x, y):
    """
    Determines if the given coordinates (x, y) are within the board bounds.

    :param x: The x-coordinate.
    :param y: The y-coordinate.
    :return: True if the coordinates are within the board bounds, False otherwise.
    """
    return 0 <= x < 8 and 0 <= y < 8

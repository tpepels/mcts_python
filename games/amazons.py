import math
from collections import deque
from games.gamestate import GameState

# Constants
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
N = 10


def visualize_amazons(state, characters=True):
    board = state.board
    output = ""

    if characters:
        cell_representation = {1: "W", 2: "B", -1: "-", 0: "."}
    else:
        cell_representation = {1: "1", 2: "2", -1: "-1", 0: "0"}
        output += "["

    for i in range(10):
        row = [cell_representation[piece] for piece in board[i]]
        if not characters:
            output += f"[{', '.join(row)}],\n"
        else:
            output += " ".join(row) + "\n"

    if not characters:
        output += "]"
    else:
        output += "\n"

    return output


class AmazonsGameState(GameState):
    def __init__(self, board=None, player=1):
        """
        Initialize the game state with the given board, player, and action.
        If no board is provided, a new one is initialized.
        """
        self.board = board if board is not None else self.initialize_board()
        self.player = player

    def initialize_board(self):
        """
        Initialize the game board with the starting positions of the Amazon pieces.
        The board is a 10x10 grid with player 1's pieces represented by 1 and player 2's pieces represented by 2.
        Empty spaces are represented by 0, and blocked spaces are represented by -1.
        """
        board = [[0] * 10 for _ in range(10)]
        board[0][3] = board[0][6] = 1
        board[9][3] = board[9][6] = 2

        board[3][9] = board[3][0] = 1
        board[6][9] = board[6][0] = 2

        return board

    def apply_action(self, action):
        """
        Apply the given action to the current game state, creating a new game state.
        An action is a tuple (x1, y1, x2, y2, x3, y3) representing the move and shot of an Amazon piece.
        """
        x1, y1, x2, y2, x3, y3 = action
        new_board = [row[:] for row in self.board]
        new_board[x2][y2] = new_board[x1][y1]  # Move the piece to the new position
        new_board[x1][y1] = 0  # Remove the piece from its old position
        new_board[x3][y3] = -1  # Block the position where the arrow was shot
        return AmazonsGameState(new_board, 3 - self.player)

    def get_legal_actions(self):
        """
        Get a list of legal actions for the current player.
        """
        actions = []
        queens = queens_from_board(self.board, self.player == 1)

        for x, y in queens:
            actions.extend(self.get_legal_moves_for_amazon(x, y))

        return actions

    def get_legal_moves_for_amazon(self, x, y):
        """
        Get a list of legal moves for the given Amazon piece at position (x, y) and the corresponding arrow shots.
        """
        moves = []

        # Iterate through all possible move directions.
        for direction in DIRECTIONS:
            dx, dy = direction
            nx, ny = x + dx, y + dy

            # Find all legal moves in the current direction.
            while 0 <= nx < N and 0 <= ny < N and self.board[nx][ny] == 0:
                arrow_shots = self.get_legal_arrow_shots(nx, ny, x, y)
                for arrow_shot in arrow_shots:
                    moves.append((x, y, nx, ny, arrow_shot[0], arrow_shot[1]))
                nx += dx
                ny += dy

        return moves

    def get_legal_arrow_shots(self, x, y, ox, oy):
        """
        Get a list of legal arrow shots for the given Amazon piece at position (x, y).
        """
        arrow_shots = []

        # Iterate through all possible shot directions.
        for direction in DIRECTIONS:
            dx, dy = direction
            nx, ny = x + dx, y + dy

            # Find all legal arrow shots in the current direction.
            while 0 <= nx < N and 0 <= ny < N and (self.board[nx][ny] == 0 or (nx == ox and ny == oy)):
                arrow_shots.append((nx, ny))
                nx += dx
                ny += dy

        return arrow_shots

    def has_legal_moves(self):
        """
        Check if the current player has any legal moves left.

        :return: True if the current player has legal moves, False otherwise.
        """
        for x in range(N):
            for y in range(N):
                if self.board[x][y] == self.player:
                    if self.get_legal_moves_for_amazon(x, y):
                        return True
        return False

    def is_terminal(self):
        """
        Check if the current game state is terminal (i.e., one player has no legal moves left).
        """
        return not self.has_legal_moves()

    def get_reward(self):
        """
        Get the reward for the current player in the current game state.

        Otherwise, the reward is 0.
        """
        if self.is_terminal():
            return 1 if self.player != 1 else -1
        return 0


# The evaluate function takes an Amazons game state and the player number (1 or 2) as input.
# It computes the number of legal moves and the number of controlled squares for each player.
# It then calculates the difference between the players for each metric and combines them with equal weights (0.5) to produce an evaluation score.


def evaluate(state, player):
    """
    Evaluate the given game state from the perspective of the specified player.

    :param state: The game state to evaluate.
    :param player: The player for whom the evaluation is being done.
    :return: A score representing the player's advantage in the game state.
    """
    opponent = 3 - player
    player_moves = 0
    opponent_moves = 0
    player_controlled_squares = 0
    opponent_controlled_squares = 0

    # Iterate through the board and collect information about player's and opponent's moves and controlled squares.
    for x in range(N):
        for y in range(N):
            piece = state.board[x][y]

            if piece == player:
                moves = state.get_legal_moves_for_amazon(x, y)
                player_moves += len(moves)
                player_controlled_squares += count_reachable_squares(state, x, y)

            elif piece == opponent:
                moves = state.get_legal_moves_for_amazon(x, y)
                opponent_moves += len(moves)
                opponent_controlled_squares += count_reachable_squares(state, x, y)

    # Calculate the differences between player's and opponent's moves and controlled squares.
    move_difference = (player_moves - opponent_moves) / (player_moves + opponent_moves)
    controlled_squares_difference = (player_controlled_squares - opponent_controlled_squares) / (
        player_controlled_squares + opponent_controlled_squares
    )

    # Return a weighted combination of the differences as the evaluation score.
    return 0.5 * move_difference + 0.5 * controlled_squares_difference


def count_reachable_squares(state, x, y):
    """
    Count the number of squares reachable by the piece at the given position (x, y) in the game state.

    :param state: The game state to evaluate.
    :param x: The x-coordinate of the piece.
    :param y: The y-coordinate of the piece.
    :return: The number of reachable squares.
    """
    reachable = 0

    # Iterate through all possible move directions.
    for direction in DIRECTIONS:
        dx, dy = direction
        nx, ny = x + dx, y + dy

        # Count the number of empty squares along the direction.
        while 0 <= nx < N and 0 <= ny < N and state.board[nx][ny] == 0:
            reachable += 1
            nx += dx
            ny += dy

    return reachable


def evaluate_lieberum(state, player):
    is_white_player = player == 1
    board = state.board
    max_depth = math.inf
    my_queen_positions = queens_from_board(board, is_white_player)
    their_queen_positions = queens_from_board(board, not is_white_player)

    # Define all the internal helper functions here
    # ... (All the other helper functions, but now without `self` and adapted to use the variables defined above)

    # Calculate the utility of the current board state using a combination of heuristics.
    utility = (
        8 * territory_heuristic(their_queen_positions, my_queen_positions, state.board, max_depth=max_depth)
        + 2 * mobility_heuristic(their_queen_positions, my_queen_positions, state.board)
        + 10 * kill_save_queens(their_queen_positions, my_queen_positions, state.board)
    )
    return utility


def kill_save_queens(their_queen_positions, my_queen_positions, board):
    """
    Calculate the kill-save heuristic, which is the difference between the number of the player's free queens
    and the number of the opponent's free queens.
    """
    total = 0
    for queen in my_queen_positions:
        free = False
        for direction in DIRECTIONS:
            new_position = generate_new_position(queen, direction)
            if is_valid_position(board, new_position):
                free = True
                break
        if free:
            total += 1

    for queen in their_queen_positions:
        free = False
        for direction in DIRECTIONS:
            new_position = generate_new_position(queen, direction)
            if is_valid_position(board, new_position):
                free = True
                break
        if free:
            total -= 1

    return total


def immediate_moves_heuristic(their_queen_positions, my_queen_positions, board):
    """
    Calculate the immediate moves heuristic, which is the difference between the number of the player's legal
    immediate moves and the number of the opponent's legal immediate moves.
    """
    total = 0
    for queen in my_queen_positions:
        for direction in DIRECTIONS:
            new_position = generate_new_position(queen, direction)
            if is_valid_position(board, new_position):
                total += 1

    for queen in their_queen_positions:
        for direction in DIRECTIONS:
            new_position = generate_new_position(queen, direction)
            if is_valid_position(board, new_position):
                total -= 1
    return total


def mobility_heuristic(their_queen_positions, my_queen_positions, board):
    """
    Calculate the mobility heuristic, which is the difference between the number of reachable squares for
    the player's queens and the number of reachable squares for the opponent's queens.
    """
    ours = [[0] * N for _ in range(N)]
    theirs = [[0] * N for _ in range(N)]

    for queen in my_queen_positions:
        old_pos = queen
        for direction in DIRECTIONS:
            new_pos = generate_new_position(old_pos, direction)
            while is_valid_position(board, new_pos):
                ours[new_pos[0]][new_pos[1]] = max(1, ours[new_pos[0]][new_pos[1]])
                new_pos = generate_new_position(new_pos, direction)

    for queen in their_queen_positions:
        old_pos = queen
        for direction in DIRECTIONS:
            new_pos = generate_new_position(old_pos, direction)
            while is_valid_position(board, new_pos):
                theirs[new_pos[0]][new_pos[1]] = max(1, theirs[new_pos[0]][new_pos[1]])
                new_pos = generate_new_position(new_pos, direction)

    return reduce_matrix(sub_matrix(ours, theirs)) + 1


def territory_heuristic(their_queen_positions, my_queen_positions, board, max_depth=math.inf):
    """
    Calculate the territory heuristic, which is the difference between the number of squares controlled by
    the player's queens and the number of squares controlled by the opponent's queens.
    """
    my_territory = [[math.inf] * N for _ in range(N)]
    their_territory = [[math.inf] * N for _ in range(N)]

    for queen in my_queen_positions:
        territory_helper(queen, board, my_territory, max_depth)

    for queen in their_queen_positions:
        territory_helper(queen, board, their_territory, max_depth)

    return territory_compare(my_territory, their_territory)


def territory_helper(queen_position, board, out, max_depth):
    """
    Update the given territory matrix 'out' with the minimum number of moves required to reach each square
    from the given queen_position.
    """
    out[queen_position[0]][queen_position[1]] = 0
    queue = deque()

    for direction in DIRECTIONS:
        new_pos = generate_new_position(queen_position, direction)
        if is_valid_position(board, new_pos):
            queue.append((direction, 1, new_pos))

    while queue:
        curr = queue.pop()
        direction, move_count, curr_pos = curr

        out[curr_pos[0]][curr_pos[1]] = min(out[curr_pos[0]][curr_pos[1]], move_count)

        if out[curr_pos[0]][curr_pos[1]] < move_count or out[curr_pos[0]][curr_pos[1]] == 0:
            continue
        if move_count >= max_depth:
            continue

        for next_direction in DIRECTIONS:
            new_pos = generate_new_position(curr_pos, next_direction)
            if not is_valid_position(board, new_pos):
                continue

            if next_direction == direction:
                queue.append((next_direction, move_count, new_pos))
            else:
                queue.append((next_direction, move_count + 1, new_pos))


def territory_compare(ours, theirs):
    """
    Compare the two given territory matrices and return the difference between the number of squares
    controlled by the player and the opponent.
    """
    return sum(
        1 if ours[row][col] < theirs[row][col] else -1 if ours[row][col] > theirs[row][col] else 0
        for row in range(N)
        for col in range(N)
    )


# Several helper functions for the game and evaluation functions


def in_bounds(row, col):
    """
    Check if the given row and col are within the bounds of the board of size N.
    """
    return 0 <= row < N and 0 <= col < N


def generate_new_position(position, direction):
    """
    Calculate a new position by adding the given direction to the current position.
    """
    return position[0] + direction[0], position[1] + direction[1]


def is_valid_position(board, position):
    """
    Check if the given position is valid on the board, i.e., it is within bounds and not blocked
    by an arrow (-1).
    """
    row, col = position
    return in_bounds(row, col) and board[row][col] != -1


def queens_from_board(board, is_white_player):
    """
    Extract the positions of the queens belonging to the current player (white or black) from the board.
    """
    return [
        (row, col) for row in range(N) for col in range(N) if board[row][col] == (1 if is_white_player else 2)
    ]


def initialize_matrix(matrix, value):
    """
    Fill the given matrix with the specified value (in-place).
    """
    for row in range(len(matrix)):
        matrix[row] = [value] * len(matrix[row])


def sub_matrix(matrix_a, matrix_b):
    """
    Subtract matrix_b from matrix_a element-wise, and return the resulting matrix.
    """
    return [[a - b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(matrix_a, matrix_b)]


def reduce_matrix(matrix):
    """
    Sum all the elements in the given matrix and return the total.
    """
    return sum([matrix[row][col] for row in range(len(matrix)) for col in range(len(matrix[row]))])

import math
import random
from collections import deque
from games.gamestate import GameState, normalize, win, loss, draw

# Constants
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class AmazonsGameState(GameState):
    players_bitstrings = [random.randint(1, 2**64 - 1) for _ in range(3)]  # 0 is for the empty player
    zobrist_tables = {
        size: [[[random.randint(1, 2**64 - 1) for _ in range(4)] for _ in range(size)] for _ in range(size)]
        for size in range(6, 11)  # Assuming anything between a 6x6 and 10x10 board
    }

    def __init__(
        self,
        board=None,
        player=1,
        board_size=10,
        n_moves=0,
        white_queens=None,
        black_queens=None,
        board_hash=None,
    ):
        """
        Initialize the game state with the given board, player, and action.
        If no board is provided, a new one is initialized.
        """
        self.board_size = board_size
        self.player = player
        self.mid = math.ceil(self.board_size / 2)
        self.n_moves = n_moves

        self.board_hash = board_hash
        self.zobrist_table = self.zobrist_tables[self.board_size]

        # Keep track of the queen positions so we can avoid having to scan the board for them
        if black_queens is None:
            self.black_queens = []
        elif board is not None:
            self.black_queens = black_queens

        if white_queens is None:
            self.white_queens = []
        elif board is not None:
            self.white_queens = white_queens

        self.board = board if board is not None else self.initialize_board()

        if self.board_hash is None:
            self.board_hash = 0
            for i in range(self.board_size):
                for j in range(self.board_size):
                    # Inititally, the board contains empty spaces (0), and four queens
                    self.board_hash ^= self.zobrist_table[i][j][self.board[i][j] % 10]
            self.board_hash ^= self.players_bitstrings[self.player]

    def initialize_board(self):
        """
        Initialize the game board with the starting positions of the Amazon pieces.
        The board is a grid with player 1's pieces represented by 1 and player 2's pieces represented by 2.
        Empty spaces are represented by 0, and blocked spaces are represented by -1.
        """

        self.white_queens = []  # 10, 11, 12, 13
        self.black_queens = []  # 20, 21, 22, 23

        board = [[0] * self.board_size for _ in range(self.board_size)]
        mid = self.board_size // 2

        # Place the black queens
        board[0][mid - 2] = 20
        self.black_queens.append((0, mid - 2))
        board[0][mid + 1] = 21
        self.black_queens.append((0, mid + 1))
        board[mid - 2][0] = 22
        self.black_queens.append((mid - 2, 0))
        board[mid - 2][self.board_size - 1] = 23
        self.black_queens.append((mid - 2, self.board_size - 1))

        # Now the white queens
        board[self.board_size - 1][mid - 2] = 10
        self.white_queens.append((self.board_size - 1, mid - 2))
        board[self.board_size - 1][mid + 1] = 11
        self.white_queens.append((self.board_size - 1, mid + 1))
        board[mid + 1][0] = 12
        self.white_queens.append((mid + 1, 0))
        board[mid + 1][self.board_size - 1] = 13
        self.white_queens.append((mid + 1, self.board_size - 1))

        return board

    def apply_action(self, action):
        """
        Apply the given action to the current game state, creating a new game state.
        An action is a tuple (x1, y1, x2, y2, x3, y3) representing the move and shot of an Amazon piece.
        """
        x1, y1, x2, y2, x3, y3 = action
        new_board = [row[:] for row in self.board]
        moving_queen = new_board[x1][y1]  # Save the moving queen number
        new_board[x2][y2] = new_board[x1][y1]  # Move the piece to the new position
        new_board[x1][y1] = 0  # Remove the piece from its old position
        new_board[x3][y3] = -1  # Block the position where the arrow was shot

        # Update board hash for the movement
        board_hash = (
            self.board_hash
            ^ self.zobrist_table[x1][y1][self.player]
            ^ self.zobrist_table[x2][y2][self.player]
            ^ self.zobrist_table[x1][y1][0]
            ^ self.zobrist_table[x3][y3][-1]
            ^ self.players_bitstrings[self.player]
            ^ self.players_bitstrings[3 - self.player]
        )
        # Copy the lists of queen positions, and update the position of the moving queen

        new_white_queens = self.white_queens.copy()
        new_black_queens = self.black_queens.copy()

        if self.player == 1 and new_white_queens is not None:
            new_white_queens[moving_queen % 10] = (x2, y2)
        elif self.player == 2 and new_black_queens is not None:
            new_black_queens[moving_queen % 10] = (x2, y2)

        return AmazonsGameState(
            new_board,
            3 - self.player,
            self.board_size,
            self.n_moves + 1,
            white_queens=new_white_queens,
            black_queens=new_black_queens,
            board_hash=board_hash,
        )

    def skip_turn(self):
        """Used for the null-move heuristic in alpha-beta search

        Returns:
            AmazonsGameState: A new gamestate in which the players are switched but no move performed
        """
        new_board = [row[:] for row in self.board]
        # Pass the same board hash since this is only used for null moves
        return AmazonsGameState(new_board, 3 - self.player, board_hash=self.board_hash)

    def get_legal_actions(self):
        """
        Get a list of legal actions for the current player.
        """
        actions = []
        queens = self.white_queens if self.player == 1 else self.black_queens

        # We start at 1 because the first position is not used
        for queen in queens:
            actions.extend(self.get_legal_moves_for_amazon(*queen))

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
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx][ny] == 0:
                # Iterate through all possible arrow shot directions.
                for arrow_direction in DIRECTIONS:
                    adx, ady = arrow_direction
                    a_nx, a_ny = nx + adx, ny + ady
                    # Find all legal arrow shots in the current direction.
                    while (0 <= a_nx < self.board_size and 0 <= a_ny < self.board_size) and (
                        self.board[a_nx][a_ny] == 0 or (a_nx == x and a_ny == y)
                    ):
                        moves.append((x, y, nx, ny, a_nx, a_ny))
                        a_nx += adx
                        a_ny += ady

                        if not (0 <= a_nx < self.board_size and 0 <= a_ny < self.board_size):
                            break
                nx += dx
                ny += dy

        return moves

    def has_legal_moves(self):
        """
        Check if the current player has any legal moves left.

        :return: True if the current player has legal moves, False otherwise.
        """
        queens = self.white_queens if self.player == 1 else self.black_queens

        for queen in queens:
            x, y = queen
            # Iterate through all possible move directions.
            for direction in DIRECTIONS:
                dx, dy = direction
                nx, ny = x + dx, y + dy

                # Check if there is a legal move in the current direction.
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx][ny] == 0:
                    return True
                nx += dx
                ny += dy

        return False

    @property
    def transposition_table_size(self):
        # return an appropriate size based on the game characteristics
        return 2**21

    def is_terminal(self):
        """
        Check if the current game state is terminal (i.e., one player has no legal moves left).
        """
        return not self.has_legal_moves()

    def get_reward(self, player):
        """
        Get the reward for the player in the current game state.

        :return: In case of a win or loss 1 or -1 otherwise, the reward is 0.
        """
        if not self.has_legal_moves():
            return win if self.player != player else loss
        return draw

    def is_capture(self, move):
        """
        Check if the given move results in a capture.

        :param move: The move to check as a tuple (from_position, to_position).
        :return: True if the move results in a capture, False otherwise.
        """

        return False

    def evaluate_move(self, move):
        """
        Evaluate the given move using a simple heuristic:
        Each move to a location closer to the center of the board is valued.
        If the move involves shooting an arrow that restricts the mobility of an opponent's Amazon, the value is increased.

        :param move: The move to evaluate.
        :return: The heuristic value of the move.
        """
        # Extract the start and end positions of the amazon and the arrow shot from the move
        _, _, end_x, end_y, arrow_x, arrow_y = move
        # Calculate score based on the distance of the Amazon's move from the center
        score = abs(self.mid - end_x) + abs(self.mid - end_y)
        # Add value to the score based on the distance of the arrow shot from the Amazon
        arrow_distance = abs(end_x - arrow_x) + abs(end_y - arrow_y)
        score += arrow_distance
        return score

    def visualize(self):
        output = ""
        cell_representation = {
            10: "W",
            11: "W",
            12: "W",
            13: "W",
            20: "B",
            21: "B",
            22: "B",
            23: "B",
            -1: "-",
            0: ".",
        }
        for i in range(self.board_size):
            row = [cell_representation[piece] for piece in self.board[i]]
            output += " ".join(row) + "\n"

        output += "hash: " + str(self.board_hash)

        return output


# The evaluate function takes an Amazons game state and the player number (1 or 2) as input.
# It computes the number of legal moves and the number of controlled squares for each player.
# It then calculates the difference between the players for each metric and combines them with equal weights (0.5) to produce an evaluation score.


def evaluate_amazons(state: AmazonsGameState, player: int, m=(0.5, 0.5), a=1, norm=True):
    """
    Evaluate the given game state from the perspective of the specified player.

    :param state: The game state to evaluate.
    :param player: The player for whom the evaluation is being done.
    :return: A score representing the player's advantage in the game state.
    """
    # Variables for the heuristics
    player_moves = 0
    opponent_moves = 0
    player_controlled_squares = 0
    opponent_controlled_squares = 0
    # The queens to iterate over
    player_queens = state.white_queens if player == 1 else state.black_queens
    opp_queens = state.white_queens if player == 2 else state.black_queens

    # Iterate through the queens and collect information about player's and opponent's moves and controlled squares.
    for queen in player_queens:
        player_moves += count_legal_moves_for_amazon(state.board, *queen)
        player_controlled_squares += count_reachable_squares(state.board, *queen)

    for queen in opp_queens:
        opponent_moves += count_legal_moves_for_amazon(state.board, *queen)
        opponent_controlled_squares += count_reachable_squares(state.board, *queen)

    # Calculate the differences between player's and opponent's moves and controlled squares.
    move_difference = (player_moves - opponent_moves) / (player_moves + opponent_moves)
    controlled_squares_difference = (player_controlled_squares - opponent_controlled_squares) / (
        player_controlled_squares + opponent_controlled_squares
    )

    # Return a weighted combination of the differences as the evaluation score.
    if norm:
        return normalize(m[0] * move_difference + m[1] * controlled_squares_difference, a)
    else:
        return m[0] * move_difference + m[1] * controlled_squares_difference


def evaluate_amazons_lieberum(
    state: AmazonsGameState, player: int, m=(0.2, 0.4, 0.3, 0.5), a=50, norm=True, n_move_cutoff=15
):
    max_depth = math.inf

    if state.n_moves < n_move_cutoff:
        return evaluate_amazons(state, player, norm=norm)

    # The queens to iterate over
    player_queens = state.white_queens if player == 1 else state.black_queens
    opp_queens = state.white_queens if player == 2 else state.black_queens

    if m[0] > 0:
        terr = territory_heuristic(opp_queens, player_queens, state.board, max_depth=max_depth)
    else:
        terr = 0

    if m[1] > 0 or m[2] > 0:
        kill_save, imm_mob = kill_save_queens_immediate_moves(opp_queens, player_queens, state.board)
    else:
        kill_save = imm_mob = 0

    if m[3] > 0:
        mob_heur = mobility_heuristic(opp_queens, player_queens, state.board)
    else:
        mob_heur = 0

    # Calculate the utility of the current board state using a combination of heuristics.
    utility = m[0] * terr + m[1] * kill_save + m[2] * imm_mob + m[3] * mob_heur

    if norm:
        return normalize(utility, a)
    else:
        return utility


def kill_save_queens_immediate_moves(their_queen_positions, my_queen_positions, board):
    """
    Calculate the kill-save heuristic, which is the difference between the number of the player's free queens
    and the number of the opponent's free queens.

    Calculate the immediate moves heuristic, which is the difference between the number of the player's legal
    immediate moves and the number of the opponent's legal immediate moves.
    """
    kill_save = 0
    imm_moves = 0
    for queen in my_queen_positions:
        count = count_reachable_squares(board, *queen)
        imm_moves += count
        if count > 0:
            kill_save += 1

    for queen in their_queen_positions:
        count = count_reachable_squares(board, *queen)
        imm_moves -= count
        if count > 0:
            kill_save -= 1

    return kill_save, imm_moves


def mobility_heuristic(their_queen_positions, my_queen_positions, board):
    """
    Calculate the mobility heuristic, which is the difference between the number of reachable squares for
    the player's queens and the number of reachable squares for the opponent's queens.
    """
    my_score = 0
    their_score = 0

    for queen in my_queen_positions:
        my_score += flood_fill(board, queen)
    for queen in their_queen_positions:
        their_score += flood_fill(board, queen)
    return my_score - their_score


def flood_fill(board, pos):
    stack = [pos]
    visited = set()

    while stack:
        x, y = stack.pop()
        visited.add((x, y))

        for direction in DIRECTIONS:
            new_pos = generate_new_position((x, y), direction)

            if can_move_to(board, new_pos) and new_pos not in visited:
                stack.append(new_pos)
    # print(f"__{pos}__{board[pos[0]][pos[1]]}__{visited}__n:{len(visited) - 1}")
    return len(visited) - 1


def territory_heuristic(their_queen_positions, my_queen_positions, board, max_depth=math.inf):
    """
    Calculate the territory heuristic, which is the difference between the number of squares controlled by
    the player's queens and the number of squares controlled by the opponent's queens.

    Note that this is an expensive method
    """
    size = len(board)
    my_territory = [[math.inf] * size for _ in range(size)]
    their_territory = [[math.inf] * size for _ in range(size)]

    for queen in my_queen_positions:
        territory_helper(queen, board, my_territory, max_depth)

    for queen in their_queen_positions:
        territory_helper(queen, board, their_territory, max_depth)

    return territory_compare(my_territory, their_territory, size)


def territory_helper(queen_position, board, out, max_depth):
    """
    Update the given territory matrix 'out' with the minimum number of moves required to reach each square
    from the given queen_position.
    """
    out[queen_position[0]][queen_position[1]] = 0
    queue = deque()

    for direction in DIRECTIONS:
        new_pos = generate_new_position(queen_position, direction)
        if can_move_to(board, new_pos):
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
            if not can_move_to(board, new_pos):
                continue

            if next_direction == direction:
                queue.append((next_direction, move_count, new_pos))
            else:
                queue.append((next_direction, move_count + 1, new_pos))


def territory_compare(ours, theirs, size):
    """
    Compare the two given territory matrices and return the difference between the number of squares
    controlled by the player and the opponent.
    """
    return sum(
        1 if ours[row][col] < theirs[row][col] else -1 if ours[row][col] > theirs[row][col] else 0
        for row in range(size)
        for col in range(size)
    )


# Several helper functions for the game and evaluation functions


def generate_new_position(position, direction):
    """
    Calculate a new position by adding the given direction to the current position.
    """
    return position[0] + direction[0], position[1] + direction[1]


def can_move_to(board, position):
    """
    Check if the given position is valid on the board, i.e., it is within bounds and not blocked
    by an arrow (-1) or another piece.
    """
    row, col = position
    return (0 <= row < len(board) and 0 <= col < len(board)) and board[row][col] == 0


def count_legal_moves_for_amazon(board, x, y):
    """
    Count the number of legal moves for the given Amazon piece at position (x, y) including the corresponding arrow shots.
    """
    count = 0
    board_size = len(board)
    # Iterate through all possible move directions.
    for direction in DIRECTIONS:
        dx, dy = direction
        nx, ny = x + dx, y + dy

        # Find all legal moves in the current direction.
        while 0 <= nx < board_size and 0 <= ny < board_size and board[nx][ny] == 0:
            # Iterate through all possible arrow shot directions.
            for arrow_direction in DIRECTIONS:
                adx, ady = arrow_direction
                a_nx, a_ny = nx + adx, ny + ady

                # Find all legal arrow shots in the current direction.
                while (
                    0 <= a_nx < board_size
                    and 0 <= a_ny < board_size
                    and (board[a_nx][a_ny] == 0 or (a_nx == x and a_ny == y))
                ):
                    count += 1
                    a_nx += adx
                    a_ny += ady

            nx += dx
            ny += dy

    return count


def count_reachable_squares(board, x, y):
    """
    Count the number of squares reachable by the piece at the given position (x, y) in the game state.

    :param state: The game state to evaluate.
    :param x: The x-coordinate of the piece.
    :param y: The y-coordinate of the piece.
    :return: The number of reachable squares.
    """
    reachable = 0
    size = len(board)
    # Iterate through all possible move directions.
    for direction in DIRECTIONS:
        dx, dy = direction
        nx, ny = x + dx, y + dy

        # Count the number of empty squares along the direction.
        while 0 <= nx < size and 0 <= ny < size and board[nx][ny] == 0:
            reachable += 1
            nx += dx
            ny += dy

    return reachable

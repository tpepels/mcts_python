import math
import random
import numpy as np
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
        board_hash=None,
    ):
        """
        Initialize the game state with the given board, player, and action.
        If no board is provided, a new one is initialized.
        """

        assert 6 <= board_size <= 10

        self.board_size = board_size
        self.player = player
        self.mid = math.ceil(self.board_size / 2)
        self.n_moves = n_moves

        self.board_hash = board_hash
        self.zobrist_table = self.zobrist_tables[self.board_size]

        self.board = board if board is not None else self.initialize_board()

        self.player_has_legal_moves = True
        if n_moves > 10:
            self.player_has_legal_moves = self.has_legal_moves()

        if self.board_hash is None:
            self.board_hash = 0
            for i in range(self.board_size):
                for j in range(self.board_size):
                    # Inititally, the board contains empty spaces (0), and four queens
                    self.board_hash ^= self.zobrist_table[i][j][self.board[i][j]]
            self.board_hash ^= self.players_bitstrings[self.player]

    def initialize_board(self):
        """
        Initialize the game board with the starting positions of the Amazon pieces.
        The board is a grid with player 1's pieces represented by 1 and player 2's pieces represented by 2.
        Empty spaces are represented by 0, and blocked spaces are represented by -1.
        """
        # Use numpy arrays for the board
        board = np.zeros((self.board_size, self.board_size), dtype=int)
        mid = self.board_size // 2
        # Place the black queens
        board[0][mid - 2] = 2
        board[0][mid + 1] = 2
        board[mid - 2][0] = 2
        board[mid - 2][self.board_size - 1] = 2
        # Now the white queens
        board[self.board_size - 1][mid - 2] = 1
        board[self.board_size - 1][mid + 1] = 1
        board[mid + 1][0] = 1
        board[mid + 1][self.board_size - 1] = 1
        return board

    def apply_action(self, action):
        """
        Apply the given action to the current game state, creating a new game state.
        An action is a tuple (x1, y1, x2, y2, x3, y3) representing the move and shot of an Amazon piece.
        """
        x1, y1, x2, y2, x3, y3 = action
        # Use numpy for the new_board
        new_board = np.copy(self.board)

        new_board[x2][y2] = new_board[x1][y1]  # Move the piece to the new position
        new_board[x1][y1] = 0  # Remove the piece from its old position
        new_board[x3][y3] = -1  # Block the position where the arrow was shot
        # Update board hash for the movement
        board_hash = (
            self.board_hash
            ^ self.zobrist_table[x1][y1][self.player]
            ^ self.zobrist_table[x2][y2][self.player]
            ^ self.zobrist_table[x3][y3][-1]
            ^ self.players_bitstrings[self.player]
            ^ self.players_bitstrings[3 - self.player]
        )
        return AmazonsGameState(
            new_board,
            3 - self.player,
            self.board_size,
            self.n_moves + 1,
            board_hash=board_hash,
        )

    def skip_turn(self):
        """Used for the null-move heuristic in alpha-beta search

        Returns:
            AmazonsGameState: A new gamestate in which the players are switched but no move performed
        """
        # Pass the same board hash since this is only used for null moves
        return AmazonsGameState(
            np.copy(self.board),
            3 - self.player,
            board_hash=self.board_hash,
            n_moves=self.n_moves + 1,
        )

    def get_legal_actions(self):
        """
        Get a list of legal actions for the current player.
        """
        queen_pos = np.where(self.board == self.player)
        queens = np.column_stack(queen_pos)
        return [move for queen in queens for move in self.get_legal_moves_for_amazon(*queen)]

    # def get_legal_moves_for_amazon(self, x, y):
    #     """
    #     Get a list of legal moves for the given Amazon piece at position (x, y) and the corresponding arrow shots.
    #     """
    #     moves = []
    #     assert self.board[x, y] > 0
    #     size = self.board.shape[0]

    #     for direction in DIRECTIONS:
    #         dx, dy = direction
    #         nx, ny = x + dx, y + dy

    #         if dx > 0:
    #             x_limit = size
    #         elif dx < 0:
    #             x_limit = -1
    #         else:
    #             x_limit = nx + 1  # unchanged

    #         if dy > 0:
    #             y_limit = size
    #         elif dy < 0:
    #             y_limit = -1
    #         else:
    #             y_limit = ny + 1  # unchanged

    #         while nx != x_limit and ny != y_limit and self.board[nx, ny] == 0:
    #             for arrow_direction in DIRECTIONS:
    #                 adx, ady = arrow_direction
    #                 a_nx, a_ny = nx + adx, ny + ady

    #                 if adx > 0:
    #                     ax_limit = size
    #                 elif adx < 0:
    #                     ax_limit = -1
    #                 else:
    #                     ax_limit = a_nx + 1  # unchanged

    #                 if ady > 0:
    #                     ay_limit = size
    #                 elif ady < 0:
    #                     ay_limit = -1
    #                 else:
    #                     ay_limit = a_ny + 1  # unchanged

    #                 while (a_nx != ax_limit and a_ny != ay_limit) and (
    #                     self.board[a_nx, a_ny] == 0 or (a_nx == x and a_ny == y)
    #                 ):
    #                     moves.append((x, y, nx, ny, a_nx, a_ny))
    #                     a_nx += adx
    #                     a_ny += ady
    #             nx += dx
    #             ny += dy

    #     return moves

    def get_legal_moves_for_amazon(self, x, y):
        """
        Get a list of legal moves for the given Amazon piece at position (x, y) and the corresponding arrow shots.
        """
        moves = []
        assert self.board[x][y] > 0
        size = self.board_size

        for direction in DIRECTIONS:
            dx, dy = direction
            nx, ny = x + dx, y + dy

            if dx > 0:
                x_limit = size
            elif dx < 0:
                x_limit = -1
            else:
                x_limit = nx + 1

            if dy > 0:
                y_limit = size
            elif dy < 0:
                y_limit = -1
            else:
                y_limit = ny + 1

            while nx != x_limit and ny != y_limit and self.board[nx][ny] == 0:
                # Find all legal arrow shots in the current direction.
                moves.extend(
                    (x, y, nx, ny, a_nx, a_ny) for a_nx, a_ny in self.generate_arrow_shots(nx, ny, x, y, size)
                )
                nx += dx
                ny += dy

        return moves

    def generate_arrow_shots(self, nx, ny, x, y, size):
        """
        Generate all legal arrow shots from the position (nx, ny).
        """
        for arrow_direction in DIRECTIONS:
            adx, ady = arrow_direction
            a_nx, a_ny = nx + adx, ny + ady

            if adx > 0:
                ax_limit = size
            elif adx < 0:
                ax_limit = -1
            else:
                ax_limit = a_nx + 1

            if ady > 0:
                ay_limit = size
            elif ady < 0:
                ay_limit = -1
            else:
                ay_limit = a_nx + 1

            while (
                a_nx != ax_limit
                and a_ny != ay_limit
                and (self.board[a_nx][a_ny] == 0 or (a_nx == x and a_ny == y))
            ):
                yield a_nx, a_ny
                a_nx += adx
                a_ny += ady

    def has_legal_moves(self):
        """
        Check if the current player has any legal moves left.

        :return: True if the current player has legal moves, False otherwise.
        """
        # If we already know the player has no legal moves, we can return False immediately
        if not self.player_has_legal_moves:
            return False

        queen_pos = np.where(self.board == self.player)
        queens = np.column_stack(queen_pos)

        size = self.board.shape[0]

        for queen in queens:
            x, y = queen

            for direction in DIRECTIONS:
                dx, dy = direction
                nx, ny = x + dx, y + dy

                if dx > 0:
                    x_limit = size
                elif dx < 0:
                    x_limit = -1
                else:
                    x_limit = nx + 1  # unchanged

                if dy > 0:
                    y_limit = size
                elif dy < 0:
                    y_limit = -1
                else:
                    y_limit = ny + 1  # unchanged

                while nx != x_limit and ny != y_limit and self.board[nx, ny] == 0:
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
        return not self.player_has_legal_moves

    def get_reward(self, player):
        """
        Get the reward for the player in the current game state.

        :return: In case of a win or loss 1 or -1 otherwise, the reward is 0.
        """
        if not self.player_has_legal_moves:
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
            1: "W",
            2: "B",
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


def evaluate_amazons(state: AmazonsGameState, player: int, m=(1.0,), a=1, norm=True):
    """
    Evaluate the given game state from the perspective of the specified player.

    :param state: The game state to evaluate.
    :param player: The player for whom the evaluation is being done.
    :return: A score representing the player's advantage in the game state.
    """
    # Variables for the heuristics
    player_controlled_squares = 0
    opponent_controlled_squares = 0

    player_queens = np.column_stack(np.where(state.board == player))
    opp_queens = np.column_stack(np.where(state.board == 3 - player))

    assert len(player_queens) == 4 and len(opp_queens) == 4
    # Iterate through the queens and collect information about player's and opponent's moves and controlled squares.
    for queen in player_queens:
        player_controlled_squares += count_reachable_squares(state.board, *queen)

    for queen in opp_queens:
        opponent_controlled_squares += count_reachable_squares(state.board, *queen)

    # Calculate the differences between player's and opponent's moves and controlled squares.
    controlled_squares_difference = (player_controlled_squares - opponent_controlled_squares) / (
        player_controlled_squares + opponent_controlled_squares
    )

    # Return a weighted combination of the differences as the evaluation score.
    if norm:
        return normalize(controlled_squares_difference, a)
    else:
        return controlled_squares_difference


def evaluate_amazons_lieberum(
    state: AmazonsGameState, player: int, m=(0.2, 0.4, 0.3, 0.5), a=50, norm=True, n_move_cutoff=15
):
    max_depth = math.inf

    if state.n_moves < n_move_cutoff:
        return evaluate_amazons(state, player, norm=norm)

    # The queens to iterate over
    player_queens = np.column_stack(np.where(state.board == player))
    opp_queens = np.column_stack(np.where(state.board == 3 - player))

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
            new_x, new_y = x + direction[0], y + direction[1]

            if (
                0 <= new_x < len(board)
                and 0 <= new_y < len(board)
                and board[new_x, new_y] == 0
                and (new_x, new_y) not in visited
            ):
                stack.append((new_x, new_y))
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
    out[queen_position[0], queen_position[1]] = 0
    queue = deque()

    for direction in DIRECTIONS:
        new_pos = np.add(queen_position, direction)
        if (
            0 <= new_pos[0] < len(board)
            and 0 <= new_pos[1] < len(board)
            and board[new_pos[0], new_pos[1]] == 0
        ):
            queue.append((direction, 1, new_pos))

    while queue:
        direction, move_count, curr_pos = queue.pop()

        if out[curr_pos[0], curr_pos[1]] < move_count or out[curr_pos[0], curr_pos[1]] == 0:
            continue
        if move_count >= max_depth:
            continue

        for next_direction in DIRECTIONS:
            new_pos = np.add(curr_pos, next_direction)
            if (
                0 <= new_pos[0] < len(board)
                and 0 <= new_pos[1] < len(board)
                and board[new_pos[0], new_pos[1]] == 0
            ):
                if next_direction == direction:
                    queue.append((next_direction, move_count, new_pos))
                else:
                    queue.append((next_direction, move_count + 1, new_pos))


def territory_compare(ours, theirs, size):
    diff_matrix = np.subtract(ours, theirs)
    return np.count_nonzero(diff_matrix < 0) - np.count_nonzero(diff_matrix > 0)


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

    for direction in DIRECTIONS:
        dx, dy = direction
        nx, ny = x + dx, y + dy

        # Precompute the limits based on the direction
        if dx > 0:
            x_limit = size
        elif dx < 0:
            x_limit = -1
        else:
            x_limit = nx + 1  # unchanged

        if dy > 0:
            y_limit = size
        elif dy < 0:
            y_limit = -1
        else:
            y_limit = ny + 1  # unchanged

        while nx != x_limit and ny != y_limit and board[nx, ny] == 0:
            reachable += 1
            nx += dx
            ny += dy

    return reachable

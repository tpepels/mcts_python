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
        white_queens=None,
        black_queens=None,
        board_hash=None,
    ):
        """
        Initialize the game state with the given board, player, and action.
        If no board is provided, a new one is initialized.
        """

        assert 6 <= board_size <= 10
        self.player_has_legal_moves = True
        self.board_size = board_size
        self.player = player
        self.mid = math.ceil(self.board_size / 2)
        self.n_moves = n_moves

        self.board_hash = board_hash
        self.zobrist_table = self.zobrist_tables[self.board_size]

        # Keep track of the queen positions so we can avoid having to scan the board for them
        if black_queens is None:
            self.black_queens = []
        else:
            self.black_queens = black_queens
            assert len(black_queens) == 4

        if white_queens is None:
            self.white_queens = []
        else:
            self.white_queens = white_queens
            assert len(white_queens) == 4

        self.board = board if board is not None else self.initialize_board()

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

        self.white_queens = []
        self.black_queens = []

        # Use numpy arrays for the board
        board = np.zeros((self.board_size, self.board_size), dtype=int)
        mid = self.board_size // 2

        # Place the black queens
        board[0][mid - 2] = 2
        self.black_queens.append((0, mid - 2))
        board[0][mid + 1] = 2
        self.black_queens.append((0, mid + 1))
        board[mid - 2][0] = 2
        self.black_queens.append((mid - 2, 0))
        board[mid - 2][self.board_size - 1] = 2
        self.black_queens.append((mid - 2, self.board_size - 1))

        # Now the white queens
        board[self.board_size - 1][mid - 2] = 1
        self.white_queens.append((self.board_size - 1, mid - 2))
        board[self.board_size - 1][mid + 1] = 1
        self.white_queens.append((self.board_size - 1, mid + 1))
        board[mid + 1][0] = 1
        self.white_queens.append((mid + 1, 0))
        board[mid + 1][self.board_size - 1] = 1
        self.white_queens.append((mid + 1, self.board_size - 1))

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
        # Copy the lists of queen positions, and update the position of the moving queen

        new_white_queens = self.white_queens.copy()
        new_black_queens = self.black_queens.copy()

        if self.player == 1:
            # Get the index of the moving queen in the list
            queen_index = new_white_queens.index((x1, y1))
            # Update the position of the queen in the list
            new_white_queens[queen_index] = (x2, y2)
        elif self.player == 2:
            # Get the index of the moving queen in the list
            queen_index = new_black_queens.index((x1, y1))
            # Update the position of the queen in the list
            new_black_queens[queen_index] = (x2, y2)

        new_state = AmazonsGameState(
            new_board,
            3 - self.player,
            self.board_size,
            self.n_moves + 1,
            white_queens=new_white_queens,
            black_queens=new_black_queens,
            board_hash=board_hash,
        )
        # Update player_has_legal_moves for the new state
        new_state.player_has_legal_moves = new_state.has_legal_moves()
        return new_state

    def skip_turn(self):
        """Used for the null-move heuristic in alpha-beta search

        Returns:
            AmazonsGameState: A new gamestate in which the players are switched but no move performed
        """
        assert self.player_has_legal_moves, "Null move should not be possible"

        # Pass the same board hash since this is only used for null moves
        return AmazonsGameState(
            np.copy(self.board),
            3 - self.player,
            board_hash=self.board_hash,
            white_queens=self.white_queens.copy(),
            black_queens=self.black_queens.copy(),
            n_moves=self.n_moves + 1,
        )

    def get_random_action(self):
        """
        Get a single random legal action for the current player.
        """
        # Select a random queen
        queens = self.white_queens if self.player == 1 else self.black_queens
        random_queen = random.choice(queens)

        # Randomly select a direction
        random_direction = random.choice(DIRECTIONS)

        move_x, move_y = random_queen[0] + random_direction[0], random_queen[1] + random_direction[1]

        num_iterations = random.randint(1, self.board_size)
        last_valid_action = None
        for _ in range(num_iterations):
            if not (
                0 <= move_x < self.board_size
                and 0 <= move_y < self.board_size
                and self.board[move_x][move_y] == 0
            ):
                break

            # Randomly select a direction for the arrow shot
            random_arrow_direction = random.choice(DIRECTIONS)
            arrow_x, arrow_y = move_x + random_arrow_direction[0], move_y + random_arrow_direction[1]

            if (
                0 <= arrow_x < self.board_size
                and 0 <= arrow_y < self.board_size
                and (
                    self.board[arrow_x][arrow_y] == 0
                    or (arrow_x == random_queen[0] and arrow_y == random_queen[1])
                )
            ):
                last_valid_action = (random_queen[0], random_queen[1], move_x, move_y, arrow_x, arrow_y)

            move_x += random_direction[0]
            move_y += random_direction[1]

        if last_valid_action is not None:
            return last_valid_action

        return self.get_random_action()

    def yield_legal_actions(self):
        """
        Get a generator of legal actions for the current player.
        """
        queens = self.white_queens if self.player == 1 else self.black_queens
        random.shuffle(queens)
        for queen in queens:
            for move in self.get_legal_moves_for_amazon(*queen):
                yield move

    def get_legal_actions(self):
        """
        Get a list of legal actions for the current player.
        """
        queens = self.white_queens if self.player == 1 else self.black_queens
        random.shuffle(queens)
        return [move for queen in queens for move in self.get_legal_moves_for_amazon(*queen)]

    def get_legal_moves_for_amazon(self, x, y):
        """
        Get a list of legal moves for the given Amazon piece at position (x, y) and the corresponding arrow shots.
        """
        moves = []
        assert self.board[x][y] > 0

        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy  # the next cell in the direction

            while 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if self.board[nx][ny] == 0:  # free cell
                    # Find all legal arrow shots in the current direction.
                    moves.extend(
                        (x, y, nx, ny, a_nx, a_ny) for a_nx, a_ny in self.generate_arrow_shots(nx, ny, x, y)
                    )
                    nx += dx
                    ny += dy  # keep exploring in the direction

                else:  # blocked cell
                    break

        return moves

    def generate_arrow_shots(self, nx, ny, x, y):
        """
        Generate all legal arrow shots from the position (nx, ny).
        """
        for adx, ady in DIRECTIONS:
            a_nx, a_ny = nx + adx, ny + ady  # the next cell in the direction

            while 0 <= a_nx < self.board_size and 0 <= a_ny < self.board_size:
                if self.board[a_nx][a_ny] == 0 or (a_nx == x and a_ny == y):  # free cell or starting cell
                    yield a_nx, a_ny

                elif self.board[a_nx][a_ny] != 0:  # blocked cell
                    break

                a_nx += adx
                a_ny += ady

    def has_legal_moves(self):
        """
        Check if the current player has any legal moves left.

        :return: True if the current player has legal moves, False otherwise.
        """
        queens = self.white_queens if self.player == 1 else self.black_queens
        for queen in queens:
            x, y = queen
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy  # the next cell in the direction

                while 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if self.board[nx][ny] == 0:  # free cell
                        return True

                    elif self.board[nx][ny] != 0:  # blocked cell
                        break

                    nx += dx
                    ny += dy

        # If no valid move is found for any queen, return False
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

    def evaluate_moves(self, moves):
        """
        Evaluate the given moves using a simple heuristic:
        Each move to a location closer to the center of the board is valued.
        If the move involves shooting an arrow that restricts the mobility of an opponent's Amazon, the value is increased.

        :param moves: The list of moves to evaluate.
        :return: The list of heuristic values of the moves.
        """
        scores = []
        for move in moves:
            # Extract the start and end positions of the amazon and the arrow shot from the move
            _, _, end_x, end_y, arrow_x, arrow_y = move
            # Calculate score based on the distance of the Amazon's move from the center
            score = abs(self.mid - end_x) + abs(self.mid - end_y)
            # Add value to the score based on the distance of the arrow shot from the Amazon
            arrow_distance = abs(end_x - arrow_x) + abs(end_y - arrow_y)
            score += arrow_distance
            scores.append((move, score))
        return scores

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
        output = "Player: " + str(self.player) + "\n"
        cell_representation = {
            1: "W",
            2: "B",
            -1: "-",
            0: ".",
        }
        for i in range(self.board_size):
            row = [cell_representation[piece] for piece in self.board[i]]
            output += " ".join(row) + "\n"

        output += "hash: " + str(self.board_hash) + "\n"
        output += "w:" + str(self.white_queens) + "\n"
        output += "b:" + str(self.black_queens) + "\n"
        output += "n_moves: " + str(self.n_moves) + " legal moves left? " + str(self.player_has_legal_moves)
        output += "\nhas_legal_moves: " + str(self.has_legal_moves())

        return output

    def __repr__(self) -> str:
        return f"amazons{self.board_size}"


def evaluate_amazons(state: AmazonsGameState, player: int, m_opp_disc=0.9, a=1, norm=False):
    """
    Evaluate the given game state from the perspective of the specified player.

    :param state: The game state to evaluate.
    :param player: The player for whom the evaluation is being done.
    :return: A score representing the player's advantage in the game state.
    """
    # Variables for the heuristics
    player_controlled_squares = 0
    opponent_controlled_squares = 0
    # The queens to iterate over
    player_queens = state.white_queens if player == 1 else state.black_queens
    opp_queens = state.white_queens if player == 2 else state.black_queens

    # Iterate through the queens and collect information about player's and opponent's moves and controlled squares.
    for queen in player_queens:
        player_controlled_squares += count_reachable_squares(state.board, *queen)

    for queen in opp_queens:
        opponent_controlled_squares += count_reachable_squares(state.board, *queen)

    # Calculate the differences between player's and opponent's moves and controlled squares.
    controlled_squares_difference = (
        (player_controlled_squares - opponent_controlled_squares)
        / (player_controlled_squares + opponent_controlled_squares)
    ) * (m_opp_disc if state.player == 3 - player else 1)

    # Return a weighted combination of the differences as the evaluation score.
    if norm:
        return normalize(controlled_squares_difference, a)
    else:
        return controlled_squares_difference


def evaluate_amazons_lieberum(
    state: AmazonsGameState,
    player: int,
    n_moves_cutoff=15,
    m_ter=0.25,
    m_kill_s=0.25,
    m_imm=0.25,
    m_mob=0.25,
    m_opp_disc=0.9,
    a=50,
    norm=False,
):
    """
    Evaluates the current game state for Game of the Amazons using Lieberum's evaluation function,
    which takes into account territory control, the potential to capture or save queens, mobility of queens,
    and the phase of the game.

    :param state: The game state to evaluate.
    :param player: The player to evaluate for (1 or 2).
    :param n_moves_cutoff: Number of moves cutoff to switch to Lieberum's evaluation.
    :param m_ter: Weight assigned to the territory control heuristic.
    :param m_kill_s: Weight assigned to the potential to capture or save queens heuristic.
    :param m_imm: Weight assigned to the immediate moves heuristic.
    :param m_mob: Weight assigned to the mobility of the queens heuristic.
    :param m_opp_disc: Multiplier for the evaluation score if it's the opponent's turn.
    :param a: Normalization factor for the evaluation score.
    :param norm: If True, the function will return a normalized evaluation score.

    :return: The evaluation score for the given player.
    """
    max_depth = math.inf

    if state.n_moves < n_moves_cutoff:
        return evaluate_amazons(state, player, m_opp_disc=m_opp_disc, norm=norm)
    # The queens to iterate over
    player_queens = state.white_queens if player == 1 else state.black_queens
    opp_queens = state.white_queens if player == 2 else state.black_queens

    if m_ter > 0:
        terr = territory_heuristic(opp_queens, player_queens, state.board, max_depth=max_depth)
    else:
        terr = 0

    if m_kill_s > 0 or m_imm > 0:
        kill_save, imm_mob = kill_save_queens_immediate_moves(opp_queens, player_queens, state.board)
    else:
        kill_save = imm_mob = 0

    if m_mob > 0:
        mob_heur = mobility_heuristic(opp_queens, player_queens, state.board)
    else:
        mob_heur = 0

    # Calculate the utility of the current board state using a combination of heuristics.
    utility = (m_ter * terr + m_kill_s * kill_save + m_imm * imm_mob + m_mob * mob_heur) * (
        m_opp_disc if state.player == 3 - player else 1
    )

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
    size = len(board)
    while stack:
        x, y = stack.pop()
        visited.add((x, y))

        for direction in DIRECTIONS:
            new_x, new_y = x + direction[0], y + direction[1]

            if (
                0 <= new_x < size
                and 0 <= new_y < size
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
    assert isinstance(board, np.ndarray), f"Expected board to be a numpy array, but got {type(board)}"

    size = len(board)

    my_territory = np.full((size, size), np.finfo(np.float64).max)
    their_territory = np.full((size, size), np.finfo(np.float64).max)

    for queen in my_queen_positions:
        territory_helper(queen, board, my_territory, max_depth)

    for queen in their_queen_positions:
        territory_helper(queen, board, their_territory, max_depth)

    return territory_compare(my_territory, their_territory)


def territory_helper(queen_position, board, out, max_depth):
    assert isinstance(board, np.ndarray), f"Expected board to be a numpy array, but got {type(board)}"

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

        if move_count >= max_depth or out[curr_pos[0], curr_pos[1]] <= move_count:
            continue

        out[curr_pos[0], curr_pos[1]] = move_count

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


def territory_compare(ours, theirs):
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

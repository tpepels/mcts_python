# cython: language_level=3, initializedcheck=False

import cython

from cython.cimports import numpy as cnp

cnp.import_array()

import math

import random
import numpy as np
from collections import deque
from cython.cimports.includes import c_shuffle, c_random, normalize, GameState, win, loss, draw

if cython.compiled:
    print("Amazons is compiled.")
else:
    print("Amazons is just a lowly interpreted script.")

# Constants
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@cython.cclass
class AmazonsGameState(GameState):
    players_bitstrings = [random.randint(1, 2**60 - 1) for _ in range(3)]  # 0 is for the empty player
    zobrist_tables = {
        size: [[[random.randint(1, 2**60 - 1) for _ in range(4)] for _ in range(size)] for _ in range(size)]
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

        # We can update these whenever we generate moves. When we apply an action then the lists are invalid
        self.n_moves_per_black_queen = {}
        self.n_moves_per_white_queen = {}
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
        board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
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
        assert self.player_has_legal_moves, "Getting or making a move should not be possible"

        queens = self.white_queens if self.player == 1 else self.black_queens

        return get_random_action(self.board, queens)

    def yield_legal_actions(self):
        """
        Get a generator of legal actions for the current player.
        """
        assert self.player_has_legal_moves, "Getting or making a move should not be possible"
        queens = self.white_queens if self.player == 1 else self.black_queens
        c_shuffle(queens)
        for queen in queens:
            # TODO It might be faster to just use the cython function here instead of the generator...
            for move in self.get_legal_moves_for_amazon(*queen):
                yield move

    def get_legal_actions(self):
        """
        Get a list of legal actions for the current player and also updates the moves per queen.
        """
        assert self.player_has_legal_moves, "Getting or making a move should not be possible"
        queens = self.white_queens if self.player == 1 else self.black_queens
        n_moves_per_queen = self.n_moves_per_white_queen if self.player == 1 else self.n_moves_per_black_queen

        legal_actions = []
        for queen in queens:
            queen_moves = list(get_legal_moves_for_amazon(queen[0], queen[1], self.board))
            legal_actions.extend(queen_moves)
            n_moves_per_queen[queen] = len(queen_moves)

        return legal_actions

    def get_legal_moves_for_amazon(self, x, y):
        """
        Get a generator of legal moves for the given Amazon piece at position (x, y) and the corresponding arrow shots.
        """
        assert self.player_has_legal_moves, "Getting or making a move should not be possible"
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy  # the next cell in the direction

            while 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if self.board[nx, ny] == 0:  # free cell
                    # Find all legal arrow shots in the current direction.
                    for a_nx, a_ny in self.generate_arrow_shots(nx, ny, x, y):
                        yield (x, y, nx, ny, a_nx, a_ny)
                    nx += dx
                    ny += dy  # keep exploring in the direction

                else:  # blocked cell
                    break

    def generate_arrow_shots(self, nx, ny, x, y):
        """
        Generate all legal arrow shots from the position (nx, ny).
        """
        assert self.player_has_legal_moves, "Getting or making a move should not be possible"
        for adx, ady in DIRECTIONS:
            a_nx, a_ny = nx + adx, ny + ady  # the next cell in the direction

            while 0 <= a_nx < self.board_size and 0 <= a_ny < self.board_size:
                if self.board[a_nx, a_ny] == 0 or (a_nx == x and a_ny == y):  # free cell or starting cell
                    yield a_nx, a_ny

                elif self.board[a_nx, a_ny] != 0:  # blocked cell
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
                    if self.board[nx, ny] == 0:  # free cell
                        return True

                    elif self.board[nx, ny] != 0:  # blocked cell
                        break

                    nx += dx
                    ny += dy

        # If no valid move is found for any queen, return False
        return False

    @property
    def transposition_table_size(self):
        # return an appropriate size based on the game characteristics
        return 2**17

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
        Moves made by queens with fewer available moves are given higher scores.

        :param moves: The list of moves to evaluate.
        :return: The list of heuristic values of the moves.
        """
        scores = [0] * len(moves)
        nmpq = self.n_moves_per_white_queen if self.player == 1 else self.n_moves_per_black_queen

        for i in range(len(moves)):
            scores[i] = (
                moves[i],
                evaluate_move(nmpq, moves[i], self.mid, self.board, self.player),
            )

        return scores

    def move_weights(self, moves):
        """
        Evaluate the given moves using a simple heuristic:
        Each move to a location closer to the center of the board is valued.
        If the move involves shooting an arrow that restricts the mobility of an opponent's Amazon, the value is increased.
        Moves made by queens with fewer available moves are given higher scores.

        :param moves: The list of moves to evaluate.
        :return: The list of heuristic values of the moves.
        """
        scores = [0] * len(moves)
        nmpq = self.n_moves_per_white_queen if self.player == 1 else self.n_moves_per_black_queen

        for i in range(len(moves)):
            scores[i] = evaluate_move(nmpq, moves[i], self.mid, self.board, self.player)

        return scores

    def evaluate_move(self, move):
        """
        Evaluate the given move using a simple heuristic:
        Each move to a location closer to the center of the board is valued.
        If the move involves shooting an arrow that restricts the mobility of an opponent's Amazon, the value is increased.

        :param move: The move to evaluate.
        :return: The heuristic value of the move.
        """
        return move, evaluate_move(
            self.n_moves_per_white_queen if self.player == 1 else self.n_moves_per_black_queen,
            move,
            self.mid,
            self.board,
            self.player,
        )

    def visualize(self, full_debug=False):
        output = "Player: " + str(self.player) + "\n"
        cell_representation = {
            1: "W",
            2: "B",
            -1: "-",
            0: ".",
        }
        output += "  " + " ".join(str(j) for j in range(self.board_size)) + "\n"  # column indices
        for i in range(self.board_size):
            row = [cell_representation[piece] for piece in self.board[i]]
            output += str(i) + " " * (2 - len(str(i))) + " ".join(row) + "\n"  # row index and the row content

        output += "hash: " + str(self.board_hash) + "\n"
        output += "w:" + str(self.white_queens) + "\n"
        output += "b:" + str(self.black_queens) + "\n"

        if full_debug:
            n_moves = 0
            if self.player_has_legal_moves:
                n_moves = len(self.get_legal_actions())
            output += f"Player: {self.player} has legal moves: {str(self.has_legal_moves())} | Turn: {self.n_moves} | # of moves: {n_moves}\n"
            output += f"Reward: {self.get_reward(1)}/{self.get_reward(2)}, Terminal: {self.is_terminal()}\n"
            output += "..." * 60 + "\n"
            output += f"Evaluation w:{evaluate_amazons(self, 1)} (b: {evaluate_amazons(self, 2)})\n"
            output += f"Lieberum Evaluation w:{evaluate_amazons_lieberum(self, 1)} (b: {evaluate_amazons_lieberum(self, 2)})\n"

            white_queens = self.white_queens
            black_queens = self.black_queens
            output += "..." * 60 + "\n"
            for queen in white_queens:
                output += f"White {queen} reachable squares: {count_reachable_squares(self.board, queen[0], queen[1])}\n"
            for queen in black_queens:
                output += f"Black {queen} reachable squares: {count_reachable_squares(self.board, queen[0], queen[1])}\n"

            output += "..." * 60 + "\n"

            output += f"White kill/save/imm: {kill_save_queens_immediate_moves(black_queens, white_queens, self.board)}\n"
            output += f"Black kill/save/imm: {kill_save_queens_immediate_moves(white_queens, black_queens, self.board)}\n"

            output += f"White mobility: {mobility_heuristic(black_queens, white_queens, self.board)}\n"
            output += f"Black mobility: {mobility_heuristic(white_queens, black_queens, self.board)}\n"

            output += f"White territory: {territory_heuristic(black_queens, white_queens, self.board, max_depth=3)}\n"
            output += f"Black territory: {territory_heuristic(white_queens, black_queens, self.board, max_depth=3)}\n"

            if n_moves > 0:
                actions = self.evaluate_moves(self.get_legal_actions())
                actions = sorted(actions, key=lambda x: x[1], reverse=True)
                output += "..." * 60 + "\n"
                output += str(actions)

        return output

    def __repr__(self) -> str:
        return "amazons" + str(self.board_size)


@cython.ccall
@cython.boundscheck(False)
@cython.returns(
    tuple[
        cython.int,
        cython.int,
        cython.int,
        cython.int,
        cython.int,
        cython.int,
    ]
)
def get_random_action(board: cython.int[:, :], queens: cython.list):
    actions: cython.list = []

    while actions == []:
        random_queen = queens[c_random(0, len(queens) - 1)]
        actions = get_legal_moves_for_amazon(random_queen[0], random_queen[1], board)

    return actions[c_random(0, len(actions) - 1)]


@cython.ccall
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.locals(dx=cython.int, dy=cython.int, nx=cython.int, ny=cython.int)
def get_legal_moves_for_amazon(x: cython.int, y: cython.int, board: cython.int[:, :]) -> cython.list:
    """
    Get a list of legal moves for the given Amazon piece at position (x, y) and the corresponding arrow shots.
    """
    moves: cython.list = []
    arrow_shots: cython.list
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy  # the next cell in the direction

            while 0 <= nx < board.shape[0] and 0 <= ny < board.shape[1]:
                if board[nx, ny] == 0:  # free cell
                    # Find all legal arrow shots in the current direction.
                    arrow_shots = generate_arrow_shots(nx, ny, x, y, board)
                    for i in range(len(arrow_shots)):
                        moves.append((x, y, nx, ny, arrow_shots[i][0], arrow_shots[i][1]))
                    nx += dx
                    ny += dy  # keep exploring in the direction

                else:  # blocked cell
                    break
    return moves


@cython.ccall
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.locals(adx=cython.int, ady=cython.int, a_nx=cython.int, a_ny=cython.int)
def generate_arrow_shots(
    nx: cython.int, ny: cython.int, x: cython.int, y: cython.int, board: cython.int[:, :]
) -> cython.list:
    """
    Generate all legal arrow shots from the position (nx, ny).
    """
    arrow_shots: cython.list = []
    for adx in range(-1, 2):
        for ady in range(-1, 2):
            if adx == 0 and ady == 0:
                continue
            a_nx, a_ny = nx + adx, ny + ady  # the next cell in the direction

            while 0 <= a_nx < board.shape[0] and 0 <= a_ny < board.shape[1]:
                if board[a_nx, a_ny] == 0 or (a_nx == x and a_ny == y):  # free cell or starting cell
                    arrow_shots.append((a_nx, a_ny))

                elif board[a_nx, a_ny] != 0:  # blocked cell
                    break

                a_nx += adx
                a_ny += ady

    return arrow_shots


@cython.ccall
@cython.infer_types(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def evaluate_move(
    n_moves_per_queen: cython.dict,
    move: tuple[
        cython.int,
        cython.int,
        cython.int,
        cython.int,
        cython.int,
        cython.int,
    ],
    mid: cython.uint,
    board: cython.int[:, :],
    player: cython.int,
) -> cython.int:
    """
    Evaluate the given move using a simple heuristic:
    Each move to a location closer to the center of the board is valued.
    If the move involves shooting an arrow that restricts the mobility of an opponent's Amazon, the value is increased.

    :param move: The move to evaluate.
    :return: The heuristic value of the move.
    """
    score: cython.int = 0

    start_x: cython.int
    start_y: cython.int
    end_x: cython.int
    end_y: cython.int
    arrow_x: cython.int
    arrow_y: cython.int
    dx: cython.int
    dy: cython.int
    ny: cython.int
    nx: cython.int

    # Extract the start and end positions of the amazon and the arrow shot from the move
    start_x, start_y, end_x, end_y, arrow_x, arrow_y = move

    # If we can throw an arrow at an opponent's queen, increase the score
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            nx, ny = arrow_x + dx, arrow_y + dy
            if 0 <= nx < board.shape[0] and 0 <= ny < board.shape[0] and board[nx, ny] == 3 - player:
                score += 40
            # Prefer moves where we still have room to move
            nx, ny = end_x + dx, end_y + dy
            if 0 <= nx < board.shape[0] and 0 <= ny < board.shape[0] and board[nx, ny] == 0:
                score += 5

    if score < 40:
        # Calculate score based on the distance of the Amazon's move from the center
        score += (mid - abs(mid - end_x)) + (mid - abs(mid - end_y))

        # Add value to the score based on the distance of the arrow shot from the Amazon
        score += abs(end_x - arrow_x) + abs(end_y - arrow_y)

    # Subtract the number of moves the queen has available (to prioritize queens with fewer moves)
    score -= int(math.log(max(1, n_moves_per_queen[(start_x, start_y)])))

    return 1000 + score  # Score cannot be negative for the roulette selection


@cython.ccall
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
def evaluate_amazons(
    state: AmazonsGameState,
    player: cython.int,
    m_opp_disc: cython.float = 0.9,
    a: cython.int = 1,
    norm: cython.bint = 0,
) -> cython.double:
    """
    Evaluate the given game state from the perspective of the specified player.

    :param state: The game state to evaluate.
    :param player: The player for whom the evaluation is being done.
    :return: A score representing the player's advantage in the game state.
    """
    # Variables for the heuristics
    player_controlled_squares: cython.double = 0
    opponent_controlled_squares: cython.double = 0
    board: cnp.ndarray = state.board

    # The queens to iterate over
    player_queens: cython.list = state.white_queens
    if player == 2:
        player_queens = state.black_queens

    opp_queens: cython.list = state.black_queens
    if player == 2:
        opp_queens = state.white_queens

    i: cython.int

    # Iterate through the queens and collect information about player's and opponent's moves and controlled squares.
    for i in range(4):
        player_controlled_squares += count_reachable_squares(board, player_queens[i][0], player_queens[i][1])
    for i in range(4):
        opponent_controlled_squares += count_reachable_squares(board, opp_queens[i][0], opp_queens[i][1])

    board_size: cython.int = board.shape[0] * board.shape[1]
    player_controlled_squares = (board_size * player_controlled_squares) / (
        player_controlled_squares**2 + 1
    )
    opponent_controlled_squares = (board_size * opponent_controlled_squares) / (
        opponent_controlled_squares**2 + 1
    )
    # Calculate the differences between player's and opponent's moves and controlled squares.
    controlled_squares_difference: cython.double = player_controlled_squares - opponent_controlled_squares

    if state.player == 3 - player:
        controlled_squares_difference *= m_opp_disc

    # Return a weighted combination of the differences as the evaluation score.
    if norm:
        return normalize(controlled_squares_difference, a)
    else:
        return controlled_squares_difference


@cython.ccall
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
@cython.locals(
    max_depth=cython.uint,
    player_queens=cython.list,
    opp_queens=cython.list,
    terr=cython.float,
    kill_save=cython.float,
    imm_mob=cython.float,
    mob_heur=cython.float,
    utility=cython.double,
)
def evaluate_amazons_lieberum(
    state: AmazonsGameState,
    player: cython.int,
    n_moves_cutoff: cython.int = 10,
    m_ter: cython.float = 2,
    m_kill_s: cython.float = 1.5,
    m_imm: cython.float = 13,
    m_mob: cython.float = 2,
    m_opp_disc: cython.float = 1,
    a: cython.uint = 50,
    norm: cython.bint = 0,
) -> cython.double:
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
    max_depth: cython.uint = 3
    board: cnp.ndarray = state.board
    if state.n_moves < n_moves_cutoff:
        return evaluate_amazons(state, player, m_opp_disc=m_opp_disc, norm=norm)
    # The queens to iterate over
    player_queens = state.white_queens if player == 1 else state.black_queens
    opp_queens = state.white_queens if player == 2 else state.black_queens

    if m_ter > 0:
        terr = territory_heuristic(opp_queens, player_queens, board, max_depth=max_depth)
    else:
        terr = 0

    if m_kill_s > 0 or m_imm > 0:
        kill_save, imm_mob = kill_save_queens_immediate_moves(opp_queens, player_queens, board)
    else:
        kill_save = imm_mob = 0

    if m_mob > 0:
        mob_heur = mobility_heuristic(opp_queens, player_queens, board)
    else:
        mob_heur = 0

    # Calculate the utility of the current board state using a combination of heuristics.
    utility: cython.double = m_ter * terr + m_kill_s * kill_save + m_imm * imm_mob + m_mob * mob_heur

    if state.player == 3 - player:
        utility *= m_opp_disc

    if norm:
        return normalize(utility, a)
    else:
        return utility


@cython.cfunc
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.locals(
    i=cython.int,
    x=cython.int,
    y=cython.int,
    count=cython.int,
    kill_save=cython.int,
    imm_moves=cython.int,
    their_queen_positions=cython.list,
    my_queen_positions=cython.list,
    board=cnp.ndarray,
)
def kill_save_queens_immediate_moves(
    their_queen_positions: cython.list, my_queen_positions: cython.list, board
) -> tuple:
    """
    Calculate the kill-save heuristic, which is the difference between the number of the player's free queens
    and the number of the opponent's free queens.

    Calculate the immediate moves heuristic, which is the difference between the number of the player's legal
    immediate moves and the number of the opponent's legal immediate moves.
    """
    kill_save: cython.int = 0
    imm_moves: cython.int = 0

    for i in range(4):
        x = my_queen_positions[i][0]
        y = my_queen_positions[i][1]
        count = count_reachable_squares(board, x, y)
        imm_moves += count
        if count > 0:
            kill_save += 1

    for i in range(4):
        x = their_queen_positions[i][0]
        y = their_queen_positions[i][1]
        count = count_reachable_squares(board, x, y)
        imm_moves -= count
        if count > 0:
            kill_save -= 1

    return kill_save, imm_moves


@cython.cfunc
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def mobility_heuristic(
    their_queen_positions: cython.list, my_queen_positions: cython.list, board: cnp.ndarray
) -> cython.int:
    """
    Calculate the mobility heuristic, which is the difference between the number of reachable squares for
    the player's queens and the number of reachable squares for the opponent's queens.
    """
    my_score: cython.int = 0
    their_score: cython.int = 0
    x: cython.int
    y: cython.int

    for i in range(4):
        x = my_queen_positions[i][0]
        y = my_queen_positions[i][1]
        my_score += flood_fill(board, x, y)
    for i in range(4):
        x = their_queen_positions[i][0]
        y = their_queen_positions[i][1]
        their_score += flood_fill(board, x, y)

    return my_score - their_score


@cython.cfunc
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.locals(
    x=cython.int,
    y=cython.int,
    new_x=cython.int,
    new_y=cython.int,
    size=cython.int,
    dr=cython.int,
    dc=cython.int,
    stack=cython.list,
)
def flood_fill(board: cython.int[:, :], x: cython.int, y: cython.int) -> cython.int:
    stack = [(x, y)]
    visited = set()
    size = board.shape[0]
    while stack:
        x, y = stack.pop()
        visited.add((x, y))

        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == dc == 0:
                    continue
                new_x, new_y = x + dr, y + dc

                if (
                    0 <= new_x < size
                    and 0 <= new_y < size
                    and board[new_x, new_y] == 0
                    and (new_x, new_y) not in visited
                ):
                    stack.append((new_x, new_y))

    return len(visited) - 1


@cython.cfunc
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
def territory_heuristic(
    their_queen_positions: cython.list,
    my_queen_positions: cython.list,
    board: cnp.ndarray,
    max_depth: cython.uint,
) -> cython.int:
    """
    Calculate the territory heuristic, which is the difference between the number of squares controlled by
    the player's queens and the number of squares controlled by the opponent's queens.

    Note that this is an expensive method
    """

    size: cython.int = board.shape[0]

    my_territory: cython.double[:, :] = np.full((size, size), np.finfo(np.float64).max)
    their_territory: cython.double[:, :] = np.full((size, size), np.finfo(np.float64).max)

    for i in range(4):
        territory_helper(my_queen_positions[i], board, my_territory, max_depth)

    for i in range(4):
        territory_helper(their_queen_positions[i], board, their_territory, max_depth)

    return territory_compare(my_territory, their_territory)


@cython.cfunc
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.locals(
    direction=tuple[cython.int, cython.int],
    curr_pos=tuple[cython.int, cython.int],
    x=cython.int,
    y=cython.int,
    size=cython.int,
    new_x=cython.int,
    new_y=cython.int,
    dc=cython.int,
    dr=cython.int,
    move_count=cython.int,
)
def territory_helper(
    queen_position: tuple[cython.int, cython.int],
    board: cython.int[:, :],
    out: cython.double[:, :],
    max_depth: cython.int,
):
    x = queen_position[0]
    y = queen_position[1]
    out[x, y] = 0
    queue = deque()
    size = board.shape[0]

    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == dc == 0:
                continue
            new_x, new_y = x + dr, y + dc
            if 0 <= new_x < size and 0 <= new_y < size and board[new_x, new_y] == 0:
                queue.append(((dr, dc), 1, (new_x, new_y)))
    while queue:
        direction, move_count, curr_pos = queue.pop()
        x = curr_pos[0]
        y = curr_pos[1]
        if move_count >= max_depth or out[x, y] <= move_count:
            continue
        out[x, y] = move_count
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == dc == 0:
                    continue
                new_x, new_y = x + dr, y + dc
                if 0 <= new_x < size and 0 <= new_y < size and board[new_x, new_y] == 0:
                    if dr == direction[0] and dc == direction[1]:
                        queue.append(((dr, dc), move_count, (new_x, new_y)))
                    else:
                        queue.append(((dr, dc), move_count + 1, (new_x, new_y)))


@cython.cfunc
def territory_compare(ours: cython.double[:, :], theirs: cython.double[:, :]) -> cython.int:
    diff_matrix: cnp.ndarray = np.subtract(ours, theirs)
    return np.count_nonzero(diff_matrix < 0) - np.count_nonzero(diff_matrix > 0)


@cython.cfunc
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.infer_types(True)
@cython.initializedcheck(False)
def count_reachable_squares(board: cython.int[:, :], x: cython.int, y: cython.int) -> cython.int:
    """
    Count the number of squares reachable by the piece at the given position (x, y) in the game state.

    :param state: The game state to evaluate.
    :param x: The x-coordinate of the piece.
    :param y: The y-coordinate of the piece.
    :return: The number of reachable squares.
    """
    reachable: cython.int = 0
    size: cython.int = board.shape[0]

    dx: cython.int
    dy: cython.int
    ny: cython.int
    nx: cython.int
    x_limit: cython.int
    y_limit: cython.int

    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == dy == 0:
                continue

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

# cython: language_level=3

import math
import random
import numpy as np

from cython.cimports import numpy as cnp

cnp.import_array()

from collections import deque
from games.gamestate import GameState, normalize, win, loss, draw

import cython

if cython.compiled:
    print("Amazons is compiled.")
else:
    print("Amazons is just a lowly interpreted script.")

# Constants
DIRECTIONS: cython.list = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

DIRECTIONS_NP: cnp.ndarray = np.array(
    [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]], dtype=np.int8
)

DIRECTIONS_view: cython.int[:, :] = DIRECTIONS_NP


cython.declare(
    players_bitstrings=cnp.ndarray,
    zobrist_table_8=cnp.ndarray,
    zobrist_table_9=cnp.ndarray,
    zobrist_table_10=cnp.ndarray,
)

players_bitstrings = np.random.randint(1, 2**42 - 1, (3), dtype=np.uint64)
zobrist_table_8 = np.random.randint(1, 2**42 - 1, (8, 8, 4), dtype=np.uint64)
zobrist_table_9 = np.random.randint(1, 2**42 - 1, (9, 9, 4), dtype=np.uint64)
zobrist_table_10 = np.random.randint(1, 2**42 - 1, (10, 10, 4), dtype=np.uint64)


@cython.cclass
class AmazonsGameState:
    player_has_legal_moves = cython.declare(cython.bint)
    board_size = cython.declare(cython.int)
    player = cython.declare(cython.int, visibility="public")
    mid = cython.declare(cython.int)
    n_moves = cython.declare(cython.int, visibility="public")

    # board_hash = cython.declare(cython.ulong)
    board_hash = cython.declare(cnp.uint64_t, visibility="public")

    n_moves_per_black_queen = cython.declare(cython.dict)
    n_moves_per_white_queen = cython.declare(cython.dict)
    black_queens = cython.declare(cython.list, visibility="public")
    white_queens = cython.declare(cython.list, visibility="public")

    zobrist_table = cython.declare(cnp.ndarray)
    board = cython.declare(cnp.ndarray, visibility="public")

    def __init__(
        self,
        board=None,
        player=1,
        board_size=10,
        n_moves=0,
        white_queens=None,
        black_queens=None,
        board_hash=0,
    ):
        """
        Initialize the game state with the given board, player, and action.
        If no board is provided, a new one is initialized.
        """

        assert 8 <= board_size <= 10

        self.player_has_legal_moves = True
        self.board_size = board_size
        self.player = player
        self.mid = math.ceil(self.board_size / 2)
        self.n_moves = n_moves

        self.board_hash = board_hash

        if self.board_size == 8:
            self.zobrist_table = zobrist_table_8
        elif self.board_size == 9:
            self.zobrist_table = zobrist_table_9
        elif self.board_size == 10:
            self.zobrist_table = zobrist_table_10

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

    @cython.ccall
    @cython.overflowcheck(False)
    def initialize_board(self) -> cnp.ndarray:
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

        if self.board_hash == 0:
            for i in range(self.board_size):
                for j in range(self.board_size):
                    # Inititally, the board contains empty spaces (0), and four queens
                    # self.board_hash ^= self.zobrist_table[i][j][board[i][j]]
                    self.board_hash = np.bitwise_xor(
                        self.board_hash, self.zobrist_table[i, j, board[i, j]], dtype=np.uint64
                    )
            self.board_hash = np.bitwise_xor(
                self.board_hash, players_bitstrings[self.player], dtype=np.uint64
            )
        return board

    @cython.ccall
    @cython.infer_types(True)
    @cython.cdivision(True)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.overflowcheck(False)
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
        board_hash = np.bitwise_xor.reduce(
            np.array(
                [
                    self.board_hash,
                    self.zobrist_table[x1, y1, self.player],
                    self.zobrist_table[x2, y2, self.player],
                    self.zobrist_table[x3, y3, 3],
                    players_bitstrings[self.player],
                    players_bitstrings[3 - self.player],
                ],
                dtype=np.uint64,
            )
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
        assert self.player_has_legal_moves, "Getting or making a move should not be possible"
        queens = self.white_queens if self.player == 1 else self.black_queens
        random.shuffle(queens)
        for queen in queens:
            for move in self.get_legal_moves_for_amazon(*queen):
                yield move

    @cython.ccall
    @cython.infer_types(True)
    @cython.cdivision(True)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.overflowcheck(False)
    def get_legal_actions(self):
        """
        Get a list of legal actions for the current player and also updates the moves per queen.
        """
        assert self.player_has_legal_moves, "Getting or making a move should not be possible"
        queens = self.white_queens if self.player == 1 else self.black_queens
        n_moves_per_queen = self.n_moves_per_white_queen if self.player == 1 else self.n_moves_per_black_queen

        legal_actions = []
        for queen in queens:
            # ! Get the legal moves for the queen (no generator)
            queen_moves = self.get_legal_moves_for_amazon(queen[0], queen[1])
            legal_actions.extend(queen_moves)
            n_moves_per_queen[queen] = len(queen_moves)

        return legal_actions

    @cython.ccall
    @cython.infer_types(True)
    @cython.cdivision(True)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.overflowcheck(False)
    def get_legal_moves_for_amazon(self, x: cython.int, y: cython.int):
        """
        Get a list of legal moves for the given Amazon piece at position (x, y) and the corresponding arrow shots.
        """
        moves = []
        assert self.player_has_legal_moves, "Getting or making a move should not be possible"
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy  # the next cell in the direction

            while 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if self.board[nx, ny] == 0:  # free cell
                    # Find all legal arrow shots in the current direction.
                    for a_nx, a_ny in self.generate_arrow_shots(nx, ny, x, y):
                        moves.append(
                            (x, y, nx, ny, a_nx, a_ny)
                        )  # Add the move to the list instead of yielding it
                    nx += dx
                    ny += dy  # keep exploring in the direction

                else:  # blocked cell
                    break
        return moves  # Finally, return the list of all moves

    # def get_legal_moves_for_amazon(self, x, y):
    #     """
    #     Get a generator of legal moves for the given Amazon piece at position (x, y) and the corresponding arrow shots.
    #     """
    #     assert self.player_has_legal_moves, "Getting or making a move should not be possible"
    #     for dx, dy in DIRECTIONS:
    #         nx, ny = x + dx, y + dy  # the next cell in the direction

    #         while 0 <= nx < self.board_size and 0 <= ny < self.board_size:
    #             if self.board[nx, ny] == 0:  # free cell
    #                 # Find all legal arrow shots in the current direction.
    #                 for a_nx, a_ny in self.generate_arrow_shots(nx, ny, x, y):
    #                     yield (x, y, nx, ny, a_nx, a_ny)
    #                 nx += dx
    #                 ny += dy  # keep exploring in the direction

    #             else:  # blocked cell
    #                 break

    @cython.cfunc
    @cython.infer_types(True)
    @cython.cdivision(True)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.overflowcheck(False)
    def generate_arrow_shots(self, nx, ny, x, y):
        """
        Generate all legal arrow shots from the position (nx, ny).
        """
        assert self.player_has_legal_moves, "Getting or making a move should not be possible"
        shots = []
        for adx, ady in DIRECTIONS:
            a_nx, a_ny = nx + adx, ny + ady  # the next cell in the direction

            while 0 <= a_nx < self.board_size and 0 <= a_ny < self.board_size:
                if self.board[a_nx, a_ny] == 0 or (a_nx == x and a_ny == y):  # free cell or starting cell
                    shots.append((a_nx, a_ny))

                elif self.board[a_nx, a_ny] != 0:  # blocked cell
                    break

                a_nx += adx
                a_ny += ady

        return shots

    # def generate_arrow_shots(self, nx, ny, x, y):
    #     """
    #     Generate all legal arrow shots from the position (nx, ny).
    #     """
    #     assert self.player_has_legal_moves, "Getting or making a move should not be possible"
    #     for adx, ady in DIRECTIONS:
    #         a_nx, a_ny = nx + adx, ny + ady  # the next cell in the direction

    #         while 0 <= a_nx < self.board_size and 0 <= a_ny < self.board_size:
    #             if self.board[a_nx, a_ny] == 0 or (a_nx == x and a_ny == y):  # free cell or starting cell
    #                 yield a_nx, a_ny

    #             elif self.board[a_nx, a_ny] != 0:  # blocked cell
    #                 break

    #             a_nx += adx
    #             a_ny += ady

    @cython.ccall
    @cython.infer_types(True)
    @cython.cdivision(True)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.overflowcheck(False)
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
        Moves made by queens with fewer available moves are given higher scores.

        :param moves: The list of moves to evaluate.
        :return: The list of heuristic values of the moves.
        """
        scores = []
        n_moves_per_queen = self.n_moves_per_white_queen if self.player == 1 else self.n_moves_per_black_queen
        for move in moves:
            score = 0
            # Extract the start and end positions of the amazon and the arrow shot from the move
            start_x, start_y, end_x, end_y, arrow_x, arrow_y = move

            # Calculate score based on the distance of the Amazon's move from the center
            score += (self.mid - abs(self.mid - end_x)) + (self.mid - abs(self.mid - end_y))

            # Add value to the score based on the distance of the arrow shot from the Amazon
            arrow_distance = abs(end_x - arrow_x) + abs(end_y - arrow_y)
            score += arrow_distance

            # Subtract the number of moves the queen has available (to prioritize queens with fewer moves)
            score -= n_moves_per_queen[(start_x, start_y)]

            # Add a bonus for ending in a position where the queen still has room to move
            # for dx, dy in DIRECTIONS:
            #     nx, ny = end_x + dx, end_y + dy
            #     if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == 0:
            #         score += 2

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
        n_moves_per_queen = self.n_moves_per_white_queen if self.player == 1 else self.n_moves_per_black_queen
        # Extract the start and end positions of the amazon and the arrow shot from the move
        start_x, start_y, end_x, end_y, arrow_x, arrow_y = move
        # Calculate score based on the distance of the Amazon's move from the center
        score = (self.mid - abs(self.mid - end_x)) + (self.mid - abs(self.mid - end_y))
        # Add value to the score based on the distance of the arrow shot from the Amazon
        arrow_distance = abs(end_x - arrow_x) + abs(end_y - arrow_y)
        score += arrow_distance
        # Subtract the number of moves the queen has available (to prioritize queens with fewer moves)
        score -= n_moves_per_queen[(start_x, start_y)]

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
        return "amazons" + str(self.board_size)


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
) -> cython.float:
    """
    Evaluate the given game state from the perspective of the specified player.

    :param state: The game state to evaluate.
    :param player: The player for whom the evaluation is being done.
    :return: A score representing the player's advantage in the game state.
    """
    # Variables for the heuristics
    player_controlled_squares: cython.float = 0
    opponent_controlled_squares: cython.float = 0

    # The queens to iterate over
    player_queens: cython.list = state.white_queens if player == 1 else state.black_queens
    opp_queens: cython.list = state.white_queens if player == 2 else state.black_queens
    i: cython.int
    # Iterate through the queens and collect information about player's and opponent's moves and controlled squares.
    for i in range(4):
        player_controlled_squares += count_reachable_squares(
            state.board, player_queens[i][0], player_queens[i][1]
        )
    # print(f"{player_controlled_squares=}")

    for i in range(4):
        opponent_controlled_squares += count_reachable_squares(
            state.board, opp_queens[i][0], opp_queens[i][1]
        )
    # print(f"{opponent_controlled_squares=}")
    # Calculate the differences between player's and opponent's moves and controlled squares.
    controlled_squares_difference = player_controlled_squares - opponent_controlled_squares * (
        m_opp_disc if state.player == 3 - player else 1
    )

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
def evaluate_amazons_lieberum(
    state: cython.object,
    player: cython.int,
    n_moves_cutoff: cython.int = 15,
    m_ter: cython.double = 0.25,
    m_kill_s: cython.double = 0.25,
    m_imm: cython.double = 0.25,
    m_mob: cython.double = 0.25,
    m_opp_disc: cython.double = 0.9,
    a: cython.int = 50,
    norm: cython.bint = False,
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
    max_depth = 3

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


@cython.cfunc
@cython.infer_types(True)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
def kill_save_queens_immediate_moves(
    their_queen_positions: cython.list, my_queen_positions: cython.list, board: cnp.ndarray
) -> tuple:
    """
    Calculate the kill-save heuristic, which is the difference between the number of the player's free queens
    and the number of the opponent's free queens.

    Calculate the immediate moves heuristic, which is the difference between the number of the player's legal
    immediate moves and the number of the opponent's legal immediate moves.
    """
    kill_save: cython.int = 0
    imm_moves: cython.int = 0
    i: cython.int
    count: cython.int

    # Iterate through the queens and collect information about player's and opponent's moves and controlled squares.
    for i in range(4):
        count = count_reachable_squares(board, my_queen_positions[i][0], my_queen_positions[i][1])
        imm_moves += count
        if count > 0:
            kill_save += 1

    for i in range(4):
        count = count_reachable_squares(board, their_queen_positions[i][0], their_queen_positions[i][1])
        imm_moves -= count
        if count > 0:
            kill_save -= 1

    return kill_save, imm_moves


@cython.cfunc
@cython.infer_types(True)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
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


@cython.cfunc
@cython.infer_types(True)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
def flood_fill(board, pos):
    stack = [pos]
    visited = set()
    size = len(board)
    max_d = 8
    while stack:
        x, y = stack.pop()
        visited.add((x, y))

        for i in range(max_d):
            dx = DIRECTIONS_view[i, 0]
            dy = DIRECTIONS_view[i, 1]

            new_x, new_y = x + dx, y + dy

            if (
                0 <= new_x < size
                and 0 <= new_y < size
                and board[new_x, new_y] == 0
                and (new_x, new_y) not in visited
            ):
                stack.append((new_x, new_y))

    return len(visited) - 1


@cython.cfunc
@cython.infer_types(True)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
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


@cython.cfunc
@cython.infer_types(True)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
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


@cython.cfunc
@cython.infer_types(True)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
def territory_compare(ours, theirs):
    diff_matrix = np.subtract(ours, theirs)
    return np.count_nonzero(diff_matrix < 0) - np.count_nonzero(diff_matrix > 0)


@cython.ccall
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
def count_reachable_squares(board: cnp.ndarray, x: cython.int, y: cython.int):
    """
    Count the number of squares reachable by the piece at the given position (x, y) in the game state.

    :param state: The game state to evaluate.
    :param x: The x-coordinate of the piece.
    :param y: The y-coordinate of the piece.
    :return: The number of reachable squares.
    """
    reachable: cython.int = 0
    size: cython.int = board.shape[0]
    max_d: cython.int = 8  # The number of directions
    dx: cython.int = 0
    dy: cython.int = 0
    nx: cython.int = 0
    ny: cython.int = 0
    y_limit: cython.int = 0
    x_limit: cython.int = 0
    max_steps: cython.int = 0

    i: cython.int = 0

    for i in range(max_d):
        dx = DIRECTIONS_view[i, 0]
        dy = DIRECTIONS_view[i, 1]

        nx, ny = x, y

        # Precompute the limits based on the direction
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

        max_steps = min(abs(nx - x_limit), abs(ny - y_limit))

        for _ in range(max_steps):
            nx += dx
            ny += dy
            if nx == x_limit or ny == y_limit or board[nx, ny] != 0:
                break

            reachable += 1

    return reachable


# def count_reachable_squares(board, x, y):
#     """
#     Count the number of squares reachable by the piece at the given position (x, y) in the game state.

#     :param state: The game state to evaluate.
#     :param x: The x-coordinate of the piece.
#     :param y: The y-coordinate of the piece.
#     :return: The number of reachable squares.
#     """
#     reachable = 0
#     size = len(board)

#     for direction in DIRECTIONS:
#         dx, dy = direction
#         nx, ny = x + dx, y + dy

#         # Precompute the limits based on the direction
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

#         while nx != x_limit and ny != y_limit and board[nx, ny] == 0:
#             reachable += 1
#             nx += dx
#             ny += dy

#     return reachable

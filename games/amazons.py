# cython: language_level=3, initializedcheck=False

import array
import cython

from cython.cimports import numpy as cnp

cnp.import_array()

import math

import random
import numpy as np
from collections import deque
from cython.cimports.includes import c_random, normalize, GameState, win, loss, draw


if cython.compiled:
    print("Amazons is compiled.")
else:
    print("Amazons is just a lowly interpreted script.")


@cython.cclass
class AmazonsGameState(GameState):
    zobrist_tables = {
        size: [[[random.randint(1, 2**60 - 1) for _ in range(4)] for _ in range(size)] for _ in range(size)]
        for size in range(6, 11)  # Assuming anything between a 6x6 and 10x10 board
    }

    # Public fields
    player = cython.declare(cython.int, visibility="public")
    board = cython.declare(cython.int[:, :], visibility="public")
    board_hash = cython.declare(cython.longlong, visibility="public")
    last_action = cython.declare(
        cython.tuple[cython.int, cython.int, cython.int, cython.int, cython.int, cython.int],
        visibility="public",
    )
    white_queens = cython.declare(cython.list, visibility="public")
    black_queens = cython.declare(cython.list, visibility="public")
    # Private fields
    n_moves_per_queen: cython.dict

    player_has_legal_moves: cython.bint
    board_size: cython.int
    mid: cython.int
    n_moves: cython.int
    zobrist_table: cython.list

    def __init__(
        self,
        board=None,
        player=1,
        board_size=10,
        n_moves=0,
        white_queens=None,
        black_queens=None,
        board_hash=None,
        last_action=(),
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
        # We can update these whenever we generate moves. When we apply an action then the lists are invalid
        self.n_moves_per_queen = {}
        self.zobrist_table = self.zobrist_tables[self.board_size]

        if board is not None:
            self.board_hash = board_hash
            self.black_queens = black_queens
            self.white_queens = white_queens
            self.last_action = last_action
        else:
            self.board = self.initialize_board()
            self.board_hash = 0
            self.last_action = (-1, -1, -1, -1, -1, -1)
            for i in range(self.board_size):
                for j in range(self.board_size):
                    # Inititally, the board contains empty spaces (0), and four queens
                    self.board_hash ^= self.zobrist_table[i][j][self.board[i][j]]

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

    @cython.cfunc
    @cython.locals(x1=cython.int, y1=cython.int, x2=cython.int, y2=cython.int, x3=cython.int, y3=cython.int)
    @cython.returns(cython.void)
    def apply_action_playout(self, action: cython.tuple):
        x1, y1, x2, y2, x3, y3 = action

        self.board[x2, y2] = self.board[x1, y1]  # Move the piece to the new position
        self.board[x1, y1] = 0  # Remove the piece from its old position
        self.board[x3, y3] = -1  # Block the position where the arrow was shot

        if self.player == 1:
            # Get the index of the moving queen in the list
            queen_index: cython.int = self.white_queens.index((x1, y1))
            # Update the position of the queen in the list
            self.white_queens[queen_index] = (x2, y2)
        elif self.player == 2:
            # Get the index of the moving queen in the list
            queen_index: cython.int = self.black_queens.index((x1, y1))
            # Update the position of the queen in the list
            self.black_queens[queen_index] = (x2, y2)

        self.player = 3 - self.player
        self.player_has_legal_moves = self.has_legal_moves()
        self.last_action = action
        self.n_moves += 1

    @cython.ccall
    @cython.locals(x1=cython.int, y1=cython.int, x2=cython.int, y2=cython.int, x3=cython.int, y3=cython.int)
    def apply_action(self, action: cython.tuple) -> AmazonsGameState:
        """
        Apply the given action to the current game state, creating a new game state.
        An action is a tuple (x1, y1, x2, y2, x3, y3) representing the move and shot of an Amazon piece.
        """
        x1, y1, x2, y2, x3, y3 = action
        # Use numpy for the new_board
        new_board: cython.int[:, :] = np.copy(self.board)

        new_board[x2, y2] = new_board[x1, y1]  # Move the piece to the new position
        new_board[x1][y1] = 0  # Remove the piece from its old position
        new_board[x3][y3] = -1  # Block the position where the arrow was shot

        # Update board hash for the movement
        board_hash = (
            self.board_hash
            ^ self.zobrist_table[x1][y1][self.player]
            ^ self.zobrist_table[x2][y2][self.player]
            ^ self.zobrist_table[x3][y3][3]  # 3 is the arrow (to prevent wraparaound)
        )

        # Copy the lists of queen positions, and update the position of the moving queen
        new_white_queens: cython.list = self.white_queens.copy()
        new_black_queens: cython.list = self.black_queens.copy()

        if self.player == 1:
            # Get the index of the moving queen in the list
            queen_index: cython.int = new_white_queens.index((x1, y1))
            # Update the position of the queen in the list
            new_white_queens[queen_index] = (x2, y2)
        elif self.player == 2:
            # Get the index of the moving queen in the list
            queen_index: cython.int = new_black_queens.index((x1, y1))
            # Update the position of the queen in the list
            new_black_queens[queen_index] = (x2, y2)

        new_state = AmazonsGameState(
            board=new_board,
            player=3 - self.player,
            board_size=self.board_size,
            n_moves=self.n_moves + 1,
            white_queens=new_white_queens,
            black_queens=new_black_queens,
            board_hash=board_hash,
            last_action=action,
        )
        # Update player_has_legal_moves for the new state
        new_state.player_has_legal_moves = new_state.has_legal_moves()
        return new_state

    @cython.ccall
    def skip_turn(self) -> AmazonsGameState:
        """Used for the null-move heuristic in alpha-beta search

        Returns:
            AmazonsGameState: A new gamestate in which the players are switched but no move performed
        """
        assert self.player_has_legal_moves, "Null move should not be possible"

        # Pass the same board hash since this is only used for null moves
        return AmazonsGameState(
            board=self.board.copy(),
            player=3 - self.player,
            board_size=self.board_size,
            board_hash=self.board_hash,
            white_queens=self.white_queens.copy(),
            black_queens=self.black_queens.copy(),
            n_moves=self.n_moves + 1,
            last_action=self.last_action,
        )

    @cython.cfunc
    def get_random_action(self) -> cython.tuple:
        """
        Get a single random legal action for the current player.
        """
        assert self.player_has_legal_moves, "Getting or making a move should not be possible"

        queens: cython.list = self.white_queens if self.player == 1 else self.black_queens
        actions: cython.list = []

        while actions == []:
            random_queen: cython.tuple[cython.int, cython.int] = queens[c_random(0, len(queens) - 1)]
            actions = get_legal_moves_for_amazon(random_queen[0], random_queen[1], self.board)

        return actions[c_random(0, len(actions) - 1)]

    @cython.ccall
    def get_legal_actions(self) -> cython.list:
        """
        Get a list of legal actions for the current player and also updates the moves per queen.
        """
        assert self.player_has_legal_moves, "Getting or making a move should not be possible"

        queens: cython.list = self.white_queens if self.player == 1 else self.black_queens
        legal_actions: cython.list = []
        i: cython.int
        for i in range(4):  # There are always 4 queens on the board
            queen_moves: cython.list = get_legal_moves_for_amazon(queens[i][0], queens[i][1], self.board)
            legal_actions.extend(queen_moves)

            # TODO If we find no moves then we know the position is terminal
            self.n_moves_per_queen[queens[i]] = len(queen_moves)

        return legal_actions

    @cython.cfunc
    def has_legal_moves(self):
        """
        Check if the current player has any legal moves left.

        :return: True if the current player has legal moves, False otherwise.
        """
        queens: cython.list = self.white_queens if self.player == 1 else self.black_queens
        i: cython.int
        size: cython.int = self.board.shape[0]
        dx: cython.int
        dy: cython.int
        ny: cython.int
        nx: cython.int
        x_limit: cython.int
        y_limit: cython.int

        for i in range(4):
            x = queens[i][0]
            y = queens[i][1]

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

                    # TODO Hier was je gebleven segmentation fault hiero
                    if nx != x_limit and ny != y_limit:
                        if self.board[nx, ny] == 0:
                            return True

        # If no valid move is found for any queen, return False
        return False

    @property
    def transposition_table_size(self):
        # return an appropriate size based on the game characteristics
        return 2**17

    @cython.ccall
    def is_terminal(self) -> cython.bint:
        """
        Check if the current game state is terminal (i.e., one player has no legal moves left).
        """
        return not self.player_has_legal_moves

    @cython.ccall
    def get_reward(self, player: cython.int) -> cython.int:
        """
        Get the reward for the player in the current game state.

        :return: In case of a win or loss 1 or -1 otherwise, the reward is 0.
        """
        if not self.player_has_legal_moves:
            return win if self.player != player else loss
        return draw

    @cython.cfunc
    def get_result_tuple(self) -> cython.tuple:
        if not self.player_has_legal_moves:  # The player who has no more legal moves loses
            if self.player == 1:
                return (0.0, 1.0)
            elif self.player == 2:
                return (1.0, 1.0)

        return (0.5, 0.5)

    @cython.cfunc
    def is_capture(self, move: cython.tuple) -> cython.bint:
        """
        Check if the given move results in a capture.

        :param move: The move to check as a tuple (from_position, to_position).
        :return: True if the move results in a capture, False otherwise.
        """

        return False

    # These dictionaries are used by run_games to set the parameter values
    param_order: dict = {
        "n_moves_cutoff": 0,
        "m_ter": 1,
        "m_kill_s": 2,
        "m_imm": 3,
        "m_mob": 4,
        "m_opp_disc": 5,
        "a": 6,
    }

    default_params = array.array("d", [10, 2.0, 1.5, 13.0, 2.0, 1.0, 50.0])

    @cython.cfunc
    def evaluate(
        self,
        player: cython.int,
        params: cython.double[:],
        norm: cython.bint = 0,
    ) -> cython.double:
        """
        Evaluates the current game state for Game of the Amazons using Lieberum's evaluation function,
        which takes into account territory control, the potential to capture or save queens, mobility of queens,
        and the phase of the game.

        :return: The evaluation score for the given player.
        """

        # "n_moves_cutoff": 0,
        # "m_ter": 1,
        # "m_kill_s": 2,
        # "m_imm": 3,
        # "m_mob": 4,
        # "m_opp_disc": 5,
        # "a": 6,

        if self.n_moves < params[0]:
            return evaluate_amazons(self, player, m_opp_disc=params[5], norm=norm)

        # The queens to iterate over
        player_queens: cython.list = self.white_queens if player == 1 else self.black_queens
        opp_queens: cython.list = self.white_queens if player == 2 else self.black_queens

        if params[1] > 0:
            max_depth: cython.uint = 3  # TODO No idea what this does for the performance
            terr: cython.int = territory_heuristic(opp_queens, player_queens, self.board, max_depth=max_depth)
        else:
            terr: cython.int = 0

        kill_save: cython.double
        imm_mob: cython.double
        if params[2] > 0 or params[3] > 0:
            kill_save, imm_mob = kill_save_queens_immediate_moves(opp_queens, player_queens, self.board)
        else:
            kill_save = imm_mob = 0

        if params[4] > 0:
            mob_heur: cython.int = mobility_heuristic(opp_queens, player_queens, self.board)
        else:
            mob_heur: cython.int = 0

        # Calculate the utility of the current board state using a combination of heuristics.
        utility: cython.double = (
            params[1] * terr + params[2] * kill_save + params[3] * imm_mob + params[4] * mob_heur
        )

        if self.player == 3 - player:
            utility *= params[5]

        if norm:
            return normalize(utility, params[6])
        else:
            return utility

    @cython.cfunc
    def evaluate_moves(self, moves: cython.list) -> cython.list:
        """
        Evaluate the given moves using a simple heuristic:
        Each move to a location closer to the center of the board is valued.
        If the move involves shooting an arrow that restricts the mobility of an opponent's Amazon, the value is increased.
        Moves made by queens with fewer available moves are given higher scores.

        :param moves: The list of moves to evaluate.
        :return: The list of heuristic values of the moves.
        """
        scores: cython.list = [0] * len(moves)
        i: cython.int
        for i in range(len(moves)):
            scores[i] = (moves[i], self.evaluate_move(moves[i]))

        return scores

    @cython.cfunc
    def move_weights(self, moves: cython.list) -> cython.list:
        """
        Evaluate the given moves using a simple heuristic:
        Each move to a location closer to the center of the board is valued.
        If the move involves shooting an arrow that restricts the mobility of an opponent's Amazon, the value is increased.
        Moves made by queens with fewer available moves are given higher scores.

        :param moves: The list of moves to evaluate.
        :return: The list of heuristic values of the moves.
        """
        scores: cython.list = [0] * len(moves)

        i: cython.int
        for i in range(len(moves)):
            scores[i] = self.evaluate_move(moves[i])

        return scores

    @cython.cfunc
    def evaluate_move(self, move: cython.tuple) -> cython.int:
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
                if (
                    0 <= nx < self.board.shape[0]
                    and 0 <= ny < self.board.shape[0]
                    and self.board[nx, ny] == 3 - self.player
                ):
                    score += 40
                # Prefer moves where we still have room to move
                nx, ny = end_x + dx, end_y + dy
                if (
                    0 <= nx < self.board.shape[0]
                    and 0 <= ny < self.board.shape[0]
                    and self.board[nx, ny] == 0
                ):
                    score += 5

        if score < 40:
            # Calculate score based on the distance of the Amazon's move from the center
            score += (self.mid - abs(self.mid - end_x)) + (self.mid - abs(self.mid - end_y))

            # Add value to the score based on the distance of the arrow shot from the Amazon
            score += abs(end_x - arrow_x) + abs(end_y - arrow_y)

        # Subtract the number of moves the queen has available (to prioritize queens with fewer moves)
        score -= int(math.log(max(1, self.n_moves_per_queen[(start_x, start_y)])))

        return 1000 + score  # Score cannot be negative for the roulette selection

    @cython.ccall
    def visualize(self, full_debug=False):
        output = "Player: " + str(self.player) + "\n"
        cell_representation = {
            1: "W",
            2: "B",
            -1: "-",
            0: ".",
        }
        output += "  "
        for j in range(self.board_size):
            output += str(j) + " "
        output += "\n"  # column indices

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
            output += f"evaluation: {self.evaluate(1, norm=False, params=self.default_params):.3f}/{self.evaluate(2, norm=False, params=self.default_params):.3f}"
            output += f" | normalized: {self.evaluate(1, norm=True, params=self.default_params):.4f}/{self.evaluate(2, norm=True, params=self.default_params):.4f}\n"

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

            # if n_moves > 0:
            #     actions = self.evaluate_moves(self.get_legal_actions())
            #     actions = sorted(actions, key=lambda x: x[1], reverse=True)
            #     output += "..." * 60 + "\n"
            #     output += str(actions)

        return output

    def __repr__(self) -> str:
        return "amazons" + str(self.board_size)


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
    board: cython.int[:, :] = state.board

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
    board=cython.int[:, :],
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
    their_queen_positions: cython.list, my_queen_positions: cython.list, board: cython.int[:, :]
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
    board: cython.int[:, :],
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
@cython.inline
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

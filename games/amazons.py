# cython: language_level=3, wraparound=False
# distutils: language=c++

import array
import cython

from cython.cimports import numpy as cnp
from cython.cimports.cython.view import array as cvarray

cnp.import_array()

import math

import random
import numpy as np
from collections import deque
from cython.cimports.libc.math import log
from cython.cimports.libcpp.vector import vector
from cython.cimports.libcpp.stack import stack
from cython.cimports.libcpp.set import set as cset
from cython.cimports.includes import c_random, normalize, GameState, win, loss, draw, f_index


if cython.compiled:
    print("Amazons is compiled.")
else:
    print("Amazons is just a lowly interpreted script.")


@cython.cclass
class AmazonsGameState(GameState):
    zobrist_tables = {
        size: [[random.randint(1, 2**60 - 1) for _ in range(4)] for _ in range(size**2)]
        for size in range(6, 11)  # Assuming anything between a 6x6 and 10x10 board
    }

    # Public fields
    player = cython.declare(cython.int, visibility="public")
    board = cython.declare(cython.int[:], visibility="public")
    board_hash = cython.declare(cython.longlong, visibility="public")
    last_action = cython.declare(cython.tuple[cython.int, cython.int, cython.int], visibility="public")
    queens = cython.declare(cython.int[:, :], visibility="public")
    # white_queens = cython.declare(cython.list, visibility="public")
    # black_queens = cython.declare(cython.list, visibility="public")
    # Private fields
    n_moves_per_queen: cython.int[:]

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
        queens=None,
        board_hash=None,
        last_action=(0, 0, 0),
    ):
        """
        Initialize the game state with the given board, player, and action.
        If no board is provided, a new one is initialized.
        """
        assert 6 <= board_size <= 10

        self.player_has_legal_moves = 1
        self.board_size = board_size
        self.player = player
        self.mid = math.ceil(self.board_size / 2)
        self.n_moves = n_moves
        # We can update these whenever we generate moves. When we apply an action then the lists are invalid
        self.zobrist_table = self.zobrist_tables[self.board_size]
        self.n_moves_per_queen = np.zeros(board_size**2, dtype=np.int32)

        if board is not None:
            self.board_hash = board_hash
            self.queens = queens
            self.last_action = last_action
            self.board = board
        else:
            self.board = self.initialize_board()
            self.last_action = (-1, -1, -1)
            self.board_hash = 0
            for i in range(self.board_size**2):
                # Inititally, the board contains empty spaces (0), and four queens
                self.board_hash ^= self.zobrist_table[i][self.board[i]]

    @cython.ccall
    def initialize_board(self):
        """
        Initialize the game board with the starting positions of the Amazon pieces.
        The board is a grid with player 1's pieces represented by 1 and player 2's pieces represented by 2.
        Empty spaces are represented by 0, and blocked spaces are represented by -1.
        """
        # Use numpy arrays for the board
        board: cython.int[:] = np.zeros(self.board_size**2, dtype=np.int32)
        self.queens = np.zeros((2, 4), dtype=np.int32)
        s: cython.int = self.board_size
        mid: cython.int = s // 2

        # Place the black queens
        index = (0 * s) + (mid - 2)
        board[index] = 2
        self.queens[1][0] = index

        index = (0 * s) + (mid + 1)
        board[index] = 2
        self.queens[1][1] = index

        index = ((mid - 2) * s) + 0
        board[index] = 2
        self.queens[1][2] = index

        index = ((mid - 2) * s) + (s - 1)
        board[index] = 2
        self.queens[1][3] = index

        # Now the white queens
        index = ((s - 1) * s) + (mid - 2)
        board[index] = 1
        self.queens[0][0] = index

        index = ((s - 1) * s) + (mid + 1)
        board[index] = 1
        self.queens[0][1] = index

        index = ((mid + 1) * s) + 0
        board[index] = 1
        self.queens[0][2] = index

        index = ((mid + 1) * s) + (s - 1)
        board[index] = 1
        self.queens[0][3] = index

        return board

    @cython.cfunc
    @cython.locals(f_p=cython.int, t_p=cython.int, t_a=cython.int)
    @cython.returns(cython.void)
    def apply_action_playout(self, action: cython.tuple):
        f_p, t_p, t_a = action

        self.board[t_p] = self.board[f_p]  # Move the piece to the new position
        self.board[f_p] = 0  # Remove the piece from its old position
        self.board[t_a] = -1  # Block the position where the arrow was shot

        # Get the index of the moving queen in the list
        queen_index: cython.int = f_index(self.queens[self.player - 1], f_p, 4)
        # Update the position of the queen in the array
        self.queens[self.player - 1][queen_index] = t_p

        self.player = 3 - self.player
        self.player_has_legal_moves = self.has_legal_moves()
        self.last_action = action
        self.n_moves += 1

    @cython.ccall
    @cython.locals(f_p=cython.int, t_p=cython.int, t_a=cython.int)
    def apply_action(self, action: cython.tuple) -> AmazonsGameState:
        """
        Apply the given action to the current game state, creating a new game state.
        An action is a tuple (x1, y1, x2, y2, x3, y3) representing the move and shot of an Amazon piece.
        """
        f_p, t_p, t_a = action

        # Copy the lists of queen positions, and update the position of the moving queen
        new_queens: cython.int[:, :] = self.queens.copy()
        new_board: cython.int[:] = self.board.copy()

        new_board[t_p] = new_board[f_p]  # Move the piece to the new position
        new_board[f_p] = 0  # Remove the piece from its old position
        new_board[t_a] = -1  # Block the position where the arrow was shot

        # Update board hash for the movement
        board_hash = (
            self.board_hash
            ^ self.zobrist_table[f_p][self.player]
            ^ self.zobrist_table[t_p][self.player]
            ^ self.zobrist_table[t_a][3]  # 3 is the arrow (to prevent wraparaound)
        )
        # Get the index of the moving queen in the list
        queen_index: cython.int = f_index(new_queens[self.player - 1], f_p, 4)
        # Update the position of the queen in the array
        new_queens[self.player - 1][queen_index] = t_p

        new_state = AmazonsGameState(
            board=new_board,
            player=3 - self.player,
            board_size=self.board_size,
            n_moves=self.n_moves + 1,
            queens=new_queens,
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
            queens=self.queens.copy(),
            n_moves=self.n_moves + 1,
            last_action=self.last_action,
        )

    @cython.cfunc
    def get_random_action(self) -> cython.tuple:
        """
        Get a single random legal action for the current player.
        """
        assert self.player_has_legal_moves, "Getting or making a move should not be possible"
        # TODO Hier was je gebleven, je moet een enkele willekeurige actie genereren in plaats van ze allemaal
        # TODO Playouts zijn nu te traag en ze zitten teveel in deze functie te wroeten
        actions: cython.list = []
        while actions == []:
            random_queen: cython.int = self.queens[self.player - 1][c_random(0, 3)]
            actions = self.get_legal_moves_for_amazon(random_queen)

        return actions[c_random(0, len(actions) - 1)]

    @cython.ccall
    def get_legal_actions(self) -> cython.list:
        """
        Get a list of legal actions for the current player and also updates the moves per queen.
        """
        assert self.player_has_legal_moves, "Getting or making a move should not be possible"

        legal_actions: cython.list = []
        i: cython.int
        for i in range(4):  # There are always 4 queens on the board
            q_p: cython.int = self.queens[self.player - 1][i]
            queen_moves: cython.list = self.get_legal_moves_for_amazon(q_p)
            legal_actions.extend(queen_moves)

            # TODO If we find no moves then we know the position is terminal
            self.n_moves_per_queen[q_p] = len(queen_moves)

        return legal_actions

    @cython.cfunc
    def has_legal_moves(self):
        """
        Check if the current player has any legal moves left.

        :return: True if the current player has legal moves, False otherwise.
        """

        i: cython.int
        size: cython.int = self.board_size
        dx: cython.int
        dy: cython.int
        ny: cython.int
        nx: cython.int
        x_limit: cython.int
        y_limit: cython.int

        for i in range(4):
            x: cython.int = self.queens[self.player - 1][i] // size
            y: cython.int = self.queens[self.player - 1][i] % size

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

                    # For movement we only have to check whether we can move 1 step (that also means that we can fire an arrow.)
                    if (nx != x_limit and ny != y_limit) and self.board[(nx * size) + ny] == 0:
                        return 1

        # If no valid move is found for any queen, return False
        return 0

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

        return 0

    # These dictionaries are used by run_games to set the parameter values
    param_order: dict = {
        "n_moves_cutoff": 0,
        "m_ter": 1,
        "m_kill_s": 2,
        "m_imm": 3,
        "m_mob": 4,
        "m_opp_disc": 5,
        "a": 6,
        "max_depth": 7,
    }

    # default_params = array.array("d", [10, 2.0, 1.5, 13.0, 2.0, 1.0, 50.0])
    default_params = array.array("d", [0, 2.0, 1.5, 13.0, 2.0, 1.0, 300.0, 1])

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
        # "max_depth": 7

        if self.n_moves < params[0]:
            return evaluate_amazons(self, player, norm=norm, m_opp_disc=params[5])

        terr: cython.int = 0
        kill_save: cython.double = 0
        imm_mob: cython.double = 0
        mob_heur: cython.double = 0

        opp_i: cython.int = (3 - player) - 1
        my_i: cython.int = player - 1

        if params[1] > 0 and params[7] > 0:  # * Territory heuristic initialisation
            my_territory: cython.double[:] = cvarray(
                shape=(self.board.shape[0],), itemsize=cython.sizeof(cython.double), format="d"
            )
            my_territory[...] = 9999.9
            their_territory: cython.double[:] = cvarray(
                shape=(self.board.shape[0],), itemsize=cython.sizeof(cython.double), format="d"
            )
            their_territory[...] = 9999.9

        for i in range(4):
            if params[1] > 0 and params[7] > 0:  # * Territory heuristic
                territory_helper(
                    self.queens[my_i][i], self.board, my_territory, cython.cast(cython.int, params[7])
                )
                territory_helper(
                    self.queens[opp_i][i], self.board, their_territory, cython.cast(cython.int, params[7])
                )

            if params[2] > 0 or params[3] > 0:  # * Kill/Save/Immobilize heuristic
                # Player
                count: cython.int = count_reachable_squares(self.board, self.queens[my_i][i])
                imm_mob += count
                if count > 0:
                    kill_save += 1
                # Opponent
                count = count_reachable_squares(self.board, self.queens[opp_i][i])
                imm_mob -= count
                if count > 0:
                    kill_save -= 1

            if params[4] > 0:  # * Mobility heuristic
                mob_heur += flood_fill(self.board, self.queens[my_i][i])
                mob_heur -= flood_fill(self.board, self.queens[opp_i][i])

        if params[1] and params[7] > 0 > 0:  # * Territory heuristic
            terr = territory_compare(my_territory, their_territory)

        # Calculate the utility of the current board state using a combination of heuristics.
        utility: cython.double = (
            (params[1] * terr) + (params[2] * kill_save) + (params[3] * imm_mob) + (params[4] * mob_heur)
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

        end_x: cython.int
        end_y: cython.int
        arrow_x: cython.int
        arrow_y: cython.int
        new_arrow_x: cython.int
        new_arrow_y: cython.int
        new_end_x: cython.int
        new_end_y: cython.int
        arrow_adj: cython.int
        end_adj: cython.int
        dx: cython.int
        dy: cython.int
        s: cython.int = self.board_size

        # Extract the start, end positions of the amazon, and the arrow shot from the move
        start: cython.int = move[0]
        end: cython.int = move[1]
        arrow: cython.int = move[2]

        # If we can throw an arrow at an opponent's queen, increase the score
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue

                arrow_x, arrow_y = arrow // s, arrow % s
                new_arrow_x, new_arrow_y = arrow_x + dx, arrow_y + dy

                if 0 <= new_arrow_x < s and 0 <= new_arrow_y < s:
                    arrow_adj = new_arrow_x * s + new_arrow_y
                    if self.board[arrow_adj] == 3 - self.player:
                        score += 40

                # Prefer moves where we still have room to move
                end_x, end_y = end // s, end % s
                new_end_x, new_end_y = end_x + dx, end_y + dy

                if 0 <= new_end_x < s and 0 <= new_end_y < s:
                    end_adj = new_end_x * s + new_end_y
                    if self.board[end_adj] == 0:
                        score += 5

        if score < 40:
            # Calculate score based on the distance of the Amazon's move from the center
            end_x, end_y = end // s, end % s
            score += (self.mid - abs(self.mid - end_x)) + (self.mid - abs(self.mid - end_y))

            # Add value to the score based on the distance of the arrow shot from the Amazon
            arrow_x, arrow_y = arrow // s, arrow % s
            score += abs(end_x - arrow_x) + abs(end_y - arrow_y)

        # Subtract the number of moves the queen has available (to prioritize queens with fewer moves)
        score -= cython.cast(cython.int, log(max(1, self.n_moves_per_queen[start])))

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
            output += chr(65 + j) + " "  # Convert integer to corresponding uppercase letter
        output += "\n"  # column indices

        s = self.board_size
        idx: cython.int
        for i in range(s):
            row = []
            for j in range(s):
                idx = i * s + j
                piece = self.board[idx]
                row.append(cell_representation[piece])
            output += str(i) + " " * (2 - len(str(i))) + " ".join(row) + "\n"  # row index and the row content

        output += "hash: " + str(self.board_hash) + "\n"
        output += (
            "w:"
            + ", ".join([position_to_readable(self.queens[0][i], self.board_size) for i in range(4)])
            + "\n"
        )
        output += (
            "b:"
            + ", ".join([position_to_readable(self.queens[1][i], self.board_size) for i in range(4)])
            + "\n"
        )

        if full_debug:
            n_moves = 0
            if self.player_has_legal_moves:
                n_moves = len(self.get_legal_actions())

            output += f"Player: {self.player} has legal moves: {str(self.has_legal_moves())} | Turn: {self.n_moves} | # of moves: {n_moves}\n"
            output += f"Reward: {self.get_reward(1)}/{self.get_reward(2)}, Terminal: {self.is_terminal()}\n"
            output += "..." * 20 + "\n"
            output += f"evaluation: {self.evaluate(1, norm=False, params=self.default_params):.3f}/{self.evaluate(2, norm=False, params=self.default_params):.3f}"
            output += f" | normalized: {self.evaluate(1, norm=True, params=self.default_params):.4f}/{self.evaluate(2, norm=True, params=self.default_params):.4f}\n"
            output += f"evaluation: {evaluate_amazons(self, 1, norm=False):.3f}/{evaluate_amazons(self, 2, norm=False):.3f}"
            output += f" | normalized: {evaluate_amazons(self, 1, norm=True):.4f}/{evaluate_amazons(self, 2, norm=True):.4f}\n"

            output += "..." * 20 + "\n"
            for queen in self.queens[0]:
                output += f"White {position_to_readable(queen, self.board_size)} reachable squares: {count_reachable_squares(self.board, queen)}\n"
            for queen in self.queens[1]:
                output += f"Black {position_to_readable(queen, self.board_size)} reachable squares: {count_reachable_squares(self.board, queen)}\n"

            output += "..." * 20 + "\n"

            # if n_moves > 0:
            #     actions = self.evaluate_moves(self.get_legal_actions())
            #     actions = sorted(actions, key=lambda x: x[1], reverse=True)
            #     output += "..." * 60 + "\n"
            #     output += str(actions)

        return output

    def __repr__(self) -> str:
        return "amazons" + str(self.board_size)

    @cython.cfunc
    @cython.locals(dx=cython.int, dy=cython.int, nx=cython.int, ny=cython.int, i=cython.int, idx=cython.int)
    def get_legal_moves_for_amazon(self, f_p: cython.int) -> cython.list:
        """
        Get a list of legal moves for the given Amazon piece at position (x, y) and the corresponding arrow shots.
        """
        moves: cython.list = []
        s: cython.int = self.board_size
        x: cython.int = f_p // s
        y: cython.int = f_p % s
        shot: cython.int

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy  # the next cell in the direction
                while 0 <= nx < s and 0 <= ny < s:
                    idx = nx * s + ny
                    if self.board[idx] == 0:  # free cell
                        # Find all legal arrow shots in the current direction.
                        arrow_shots: vector[cython.int] = self.generate_arrow_shots(nx, ny, x, y)
                        for shot in arrow_shots:
                            moves.append((f_p, idx, shot))

                        nx += dx
                        ny += dy  # keep exploring in the direction

                    else:  # blocked cell
                        break
        return moves

    @cython.cfunc
    @cython.locals(adx=cython.int, ady=cython.int, a_nx=cython.int, a_ny=cython.int)
    def generate_arrow_shots(
        self, nx: cython.int, ny: cython.int, x: cython.int, y: cython.int
    ) -> vector[cython.int]:
        """
        Generate all legal arrow shots from the position (nx, ny).
        """

        arrow_shots: vector[cython.int]
        s: cython.int = self.board_size
        arrow_shots.reserve(max(9, (s * 6) - self.n_moves))

        for adx in range(-1, 2):
            for ady in range(-1, 2):
                if adx == 0 and ady == 0:
                    continue

                a_nx, a_ny = nx + adx, ny + ady  # the next cell in the direction

                while 0 <= a_nx < s and 0 <= a_ny < s:
                    idx = a_nx * s + a_ny
                    if self.board[idx] == 0 or (a_nx == x and a_ny == y):  # free cell or starting cell
                        arrow_shots.push_back(idx)

                    elif self.board[idx] != 0:  # blocked cell
                        break
                    a_nx += adx
                    a_ny += ady

        return arrow_shots


@cython.cfunc
def position_to_readable(pos: cython.int, board_size: cython.int) -> str:
    row = pos // board_size
    col = pos % board_size
    col_letter = chr(65 + col)  # Convert integer to corresponding uppercase letter
    return col_letter + str(row)


@cython.cfunc
def evaluate_amazons(
    state: AmazonsGameState,
    player: cython.int,
    norm: cython.bint = 0,
    m_opp_disc: cython.float = 0.9,
    a: cython.double = 20,
) -> cython.double:
    """
    Evaluate the given game state from the perspective of the specified player.

    :param state: The game state to evaluate.
    :param player: The player for whom the evaluation is being done.
    :return: A score representing the player's advantage in the game state.
    """

    p_squares: cython.double = 0
    my_i: cython.int = player - 1
    opp_i: cython.int = (3 - player) - 1
    i: cython.int

    for i in range(4):
        p_squares += count_reachable_squares(state.board, state.queens[my_i][i])
        p_squares -= count_reachable_squares(state.board, state.queens[opp_i][i])

    if not norm:
        if state.player == 3 - player:
            return p_squares * m_opp_disc
        else:
            return p_squares
    else:
        if state.player == 3 - player:
            return normalize(p_squares * m_opp_disc, a)
        else:
            return normalize(p_squares, a)


@cython.cfunc
@cython.locals(
    x=cython.int,
    y=cython.int,
    new_x=cython.int,
    new_y=cython.int,
    size=cython.int,
    dr=cython.int,
    dc=cython.int,
    s=cython.int,
    idx=cython.int,
    new_idx=cython.int,
)
def flood_fill(board: cython.int[:], pos: cython.int) -> cython.int:
    fstack: stack[cython.int]
    visited: cset[cython.int]

    fstack.push(pos)
    visited.insert(pos)

    s = board.shape[0]
    s_d: cython.int = s**2
    while not fstack.empty():
        idx = fstack.top()
        fstack.pop()

        x = idx // s
        y = idx % s

        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == dc == 0:
                    continue

                new_x, new_y = x + dr, y + dc
                new_idx = new_x * s + new_y
                if (
                    0 <= s_d < new_idx
                    and (0 <= new_x < s and 0 <= new_y < s)
                    and board[new_idx] == 0
                    and visited.count(new_idx) == 0
                ):
                    fstack.push(new_idx)
                    visited.insert(new_idx)

    return visited.size() - 1


@cython.cfunc
@cython.locals(
    direction=tuple[cython.int, cython.int],
    curr_idx=cython.int,
    idx=cython.int,
    size=cython.int,
    new_idx=cython.int,
    dc=cython.int,
    dr=cython.int,
    move_count=cython.int,
    s=cython.int,
)
def territory_helper(
    queen_idx: cython.int,
    board: cython.int[:],
    out: cython.double[:],
    max_depth: cython.int,
):
    s = int(board.shape[0] ** 0.5)  # Assuming the board is square
    out[queen_idx] = 0
    queue = deque()

    for dr in range(-s, s + 1, s):  # -s, 0, s
        for dc in range(-1, 2):  # -1, 0, 1
            if dr == dc == 0:
                continue
            new_idx = queen_idx + dr + dc
            if 0 <= new_idx < board.shape[0] and board[new_idx] == 0:
                if (dc == -1 and queen_idx % s == 0) or (dc == 1 and (queen_idx + 1) % s == 0):
                    continue  # Prevent wrap-around left or right
                queue.append(((dr, dc), 1, new_idx))

    while queue:
        direction, move_count, curr_idx = queue.pop()

        if move_count >= max_depth or out[curr_idx] <= move_count:
            continue

        out[curr_idx] = move_count
        for dr in range(-s, s + 1, s):
            for dc in range(-1, 2):
                if dr == dc == 0:
                    continue
                new_idx = curr_idx + dr + dc
                if 0 <= new_idx < board.shape[0] and board[new_idx] == 0:
                    if (dc == -1 and curr_idx % s == 0) or (dc == 1 and (curr_idx + 1) % s == 0):
                        continue  # Prevent wrap-around left or right
                    if dr == direction[0] and dc == direction[1]:
                        queue.append(((dr, dc), move_count, new_idx))
                    else:
                        queue.append(((dr, dc), move_count + 1, new_idx))


@cython.cfunc
@cython.inline
def territory_compare(ours: cython.double[:], theirs: cython.double[:]) -> cython.int:
    i: cython.int
    positive_diff: cython.int = 0
    negative_diff: cython.int = 0
    size: cython.int = ours.shape[0]

    for i in range(size):
        diff = ours[i] - theirs[i]
        if diff < 0:
            negative_diff += 1
        elif diff > 0:
            positive_diff += 1

    return negative_diff - positive_diff


@cython.cfunc
def count_reachable_squares(board: cython.int[:], pos: cython.int) -> cython.int:
    """
    Count the number of squares reachable by the piece at the given position in the game state.

    :param board: The game board in 1D representation.
    :param pos: The 1D index of the piece.
    :return: The number of reachable squares.
    """
    reachable: cython.int = 0
    size: cython.int = int(board.shape[0] ** 0.5)  # Assuming a square board

    dx: cython.int
    dy: cython.int
    ny: cython.int
    nx: cython.int
    x_limit: cython.int
    y_limit: cython.int

    x: cython.int = pos // size
    y: cython.int = pos % size

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
                x_limit = nx  # unchanged

            if dy > 0:
                y_limit = size
            elif dy < 0:
                y_limit = -1
            else:
                y_limit = ny  # unchanged

            while nx != x_limit and ny != y_limit and board[nx * size + ny] == 0:
                reachable += 1
                nx += dx
                ny += dy

    return reachable

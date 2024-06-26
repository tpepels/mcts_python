# cython: language_level=3
import array
import itertools

# from random import randint
from fastrand import pcg32randint as py_randint
import cython
import numpy as np
from cython.cimports import numpy as cnp

cnp.import_array()
from cython.cimports.includes import GameState, win, draw, loss, normalize, where_is_k2d
from cython.cimports.libc.math import pow
from cython.cimports.games.tictactoe import MIN_SIZE, MAX_SIZE
from cython.cimports.libc.limits import LONG_MAX
from termcolor import colored

MAX_SIZE = 19
MIN_SIZE = 3
# For some reason this guy always returns 1 the first time..
for i in range(0, 100):
    py_randint(0, LONG_MAX - 1)


@cython.cfunc
@cython.inline
@cython.exceptval(-99999999, check=False)
def randint(a: cython.int, b: cython.int) -> cython.int:
    return py_randint(a, b)


@cython.cclass
class TicTacToeGameState(GameState):
    zobrist_tables = {
        size: np.array(
            [[[py_randint(1, LONG_MAX - 1) for _ in range(3)] for _ in range(size)] for _ in range(size)],
            dtype=np.uint64,
        )
        for size in range(MIN_SIZE, MAX_SIZE + 1)
    }

    REUSE = True

    # Public variables
    board = cython.declare(cython.char[:, :], visibility="public")
    board_hash = cython.declare(cython.ulonglong, visibility="public")
    last_action = cython.declare(cython.tuple[cython.short, cython.short], visibility="public")
    pre_last_action = cython.declare(cython.tuple[cython.short, cython.short], visibility="public")
    longest_lines: cython.short[2]
    # Private variables
    n_moves: cython.short
    pie_move_done: cython.bint
    size: cython.short
    row_length: cython.short
    zobrist_table: cython.ulonglong[:, :, :]
    center: cython.short
    winner: cython.short

    def __init__(
        self,
        board_size: cython.short = 3,
        row_length: cython.short = 3,
        last_action: cython.tuple[cython.short, cython.short] = (-1, -1),
        pre_last_action: cython.tuple[cython.short, cython.short] = (-1, -1),
        board: cython.char[:, :] = None,
        player: cython.short = 1,
        n_moves: cython.short = 0,
        board_hash: cython.ulonglong = 0,
        pie_move_done: cython.bint = 0,
        longest_line_1: cython.short = 0,
        longest_line_2: cython.short = 0,
    ):
        # assert MIN_SIZE <= board_size <= MAX_SIZE

        self.size = board_size
        self.row_length = row_length if row_length else board_size
        self.board = board
        self.board_hash = board_hash
        self.player = player
        self.n_moves = n_moves
        self.last_action = last_action
        self.pre_last_action = pre_last_action
        self.center = self.size // 2
        self.pie_move_done = pie_move_done
        self.zobrist_table = self.zobrist_tables[self.size]

        if self.board is None:
            self.board = np.zeros((self.size, self.size), dtype=np.int8)
            self.board_hash = 0

            # Longest line per player
            self.longest_lines = np.zeros(2, dtype=np.int16)

            for i in range(self.size):
                for j in range(self.size):
                    piece = self.board[i][j]
                    self.board_hash ^= self.zobrist_table[i][j][piece]
        else:
            self.longest_lines[0] = longest_line_1
            self.longest_lines[1] = longest_line_2
            # This means that the board is not empty and we should check if the last player made a winning move
            self._check_win()

    @cython.cfunc
    @cython.locals(x=cython.short, y=cython.short)
    def apply_action_playout(self, action: cython.tuple) -> cython.void:
        x, y = action
        self.board[x, y] = self.player
        # For the pie-rule we still switch the player, because we switched the marks
        self.player = 3 - self.player
        self.pre_last_action = self.last_action
        self.last_action = action
        self.n_moves += 1
        self._check_win()

    @cython.ccall
    @cython.locals(x=cython.short, y=cython.short)
    def apply_action(self, action: cython.tuple) -> TicTacToeGameState:
        # assert len(action) == 2, f"Action should be a tuple of length 2, not {action}"

        x, y = action
        new_board: cython.char[:, :] = self.board.copy()
        move_increment: cython.short = 1
        opp: cython.short = 3 - self.player

        if self.board[x, y] != 0:
            # assert (
            #     self.n_moves == 1
            # ), "Pie Rule decision must be made after the first move!"
            # assert not self.pie_move_done, "Pie Rule decision already made!"
            # assert (
            #     self.last_action == action
            # ), f"Pie action should be the same as the first action, {self.last_action} != {action}"
            new_board[self.last_action[0], self.last_action[1]] = self.player
            move_increment = 0
            # Update the hash
            board_hash: cython.longlong = (
                self.board_hash ^ self.zobrist_table[x][y][opp] ^ self.zobrist_table[x][y][self.player]
            )
        else:
            # assert self.board[x, y] == 0, "Illegal move"
            new_board[x, y] = self.player
            # Update the hash
            board_hash: cython.longlong = (
                self.board_hash ^ self.zobrist_table[x][y][0] ^ self.zobrist_table[x][y][self.player]
            )
        # For the pie-rule we still switch the player, because we switched the marks
        return TicTacToeGameState(
            board_size=self.size,
            board=new_board,
            player=opp,
            row_length=self.row_length,
            board_hash=board_hash,
            n_moves=self.n_moves + move_increment,
            last_action=action,
            pre_last_action=self.last_action,
            pie_move_done=move_increment == 0 or self.pie_move_done,
            longest_line_1=self.longest_lines[0],
            longest_line_2=self.longest_lines[1],
        )

    @cython.cfunc
    def skip_turn(self) -> TicTacToeGameState:
        """Used for the null-move heuristic in alpha-beta search"""
        # Pass the same hash since this is only used for null-moves
        return TicTacToeGameState(
            board_size=self.size,
            board=self.board.copy(),
            player=3 - self.player,
            row_length=self.row_length,
            board_hash=self.board_hash,
            n_moves=self.n_moves,
            last_action=self.last_action,
            pre_last_action=self.last_action,
            pie_move_done=self.pie_move_done,
        )

    @cython.cfunc
    @cython.locals(
        pos=cython.short,
        x=cython.long,
        y=cython.long,
        last_x=cython.short,
        last_y=cython.short,
        last_move_int=cython.short,
        i=cython.short,
        j=cython.short,
    )
    def get_random_action(self) -> cython.tuple:
        # Trie to randomly play close to a previous move
        if self.n_moves > 2:
            roll: cython.short = randint(0, 100)
            if roll < 40 or self.longest_lines[self.player - 1] == self.row_length - 1:
                last_x = self.pre_last_action[0]
                last_y = self.pre_last_action[1]

                x = randint(0, 2) - 1 + last_x
                y = randint(0, 2) - 1 + last_y

                x = max(0, min(self.size - 1, x))
                y = max(0, min(self.size - 1, y))

                if self.board[x, y] == 0:
                    # print(":: Pre-last move: ", x, y, "::")
                    return (x, y)
            elif (roll > 40 and roll <= 70) or self.longest_lines[(3 - self.player) - 1] == self.row_length - 1:
                last_x = self.last_action[0]
                last_y = self.last_action[1]

                x = randint(0, 2) - 1 + last_x
                y = randint(0, 2) - 1 + last_y

                x = max(0, min(self.size - 1, x))
                y = max(0, min(self.size - 1, y))

                if self.board[x, y] == 0:
                    # print(":: Last move: ", x, y, "::")
                    return (x, y)

        # If we did not play close to a previous move, play near the centre
        for _ in range(randint(0, 3)):
            x = self.center + (randint(0, 4) - 2)
            y = self.center + (randint(0, 4) - 2)
            x = max(0, min(self.size - 1, x))
            y = max(0, min(self.size - 1, y))
            if self.board[x, y] == 0:
                # print(":: Centre move: ", x, y, "::")
                return (x, y)

        # If no close move could be found, return a random move
        x = randint(0, self.size - 1)
        y = randint(0, self.size - 1)
        # Loop through the board starting at the random coordinates and find the first empty space
        for i in range(self.size):
            for j in range(self.size):
                if self.board[(x + i) % self.size, (y + j) % self.size] == 0:
                    # print(f":: Random move: {x, y} ::")
                    return ((x + i) % self.size, (y + j) % self.size)

    @cython.ccall
    @cython.locals(
        i=cython.short,
        j=cython.short,
    )
    def get_legal_actions(self) -> cython.list:
        if self.player == 2 and self.n_moves == 1 and not self.pie_move_done:
            # In the case of a pie move, return all possible moves
            return [(i, j) for i in range(self.size) for j in range(self.size)]
        else:
            return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i, j] == 0]

    @cython.ccall
    def is_terminal(self) -> cython.bint:
        return self.winner != 0

    @cython.ccall
    def get_result_tuple(self) -> cython.tuple:
        if self.winner == 1:
            return (1.0, 0.0)
        elif self.winner == 2:
            return (0.0, 1.0)

        return (0.5, 0.5)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def get_reward(self, player: cython.short) -> cython.int:
        if self.winner != 0:
            if self.winner == player:
                return win
            elif self.winner == 3 - player:
                return loss
            else:
                return draw

        return 0

    @cython.ccall
    @cython.locals(
        player=cython.char,
        last_move_x=cython.short,
        last_move_y=cython.short,
        board=cython.char[:, :],
        last_player=cython.char,
        am_i_last_player=cython.bint,
        dx=cython.short,
        dy=cython.short,
        count=cython.short,
        i=cython.short,
        x=cython.short,
        y=cython.short,
    )
    def _check_win(self) -> cython.void:
        # We first need enough marks on the board to be able to win
        if self.n_moves < (self.row_length * 2) - 1:
            self.winner = 0
            return

        last_move_x = self.last_action[0]
        last_move_y = self.last_action[1]
        # only the last player to make a move can actually win the game, so we can look from their perspective
        last_player = 3 - self.player
        max_count: cython.short = 0
        # Check all 8 directions around the last move made
        for dx in range(-1, 2):
            for dy in range(0, 2):
                if (dx != 0 or dy != 0) and not (dx == -1 and dy == 0):
                    count = 1

                    # Start from the last move and check for row_length cells
                    for i in range(1, self.row_length):
                        x = last_move_x + (dx * i)
                        y = last_move_y + (dy * i)

                        # Ensure the indices are within the board
                        if 0 <= x < self.size and 0 <= y < self.size:
                            if self.board[x, y] == last_player:
                                count += 1
                                if count == self.row_length:
                                    self.winner = last_player
                                    return
                            else:
                                break
                        else:
                            break

                    for i in range(-1, -self.row_length, -1):
                        x = last_move_x + (dx * i)
                        y = last_move_y + (dy * i)

                        # Ensure the indices are within the board
                        if 0 <= x < self.size and 0 <= y < self.size:
                            if self.board[x, y] == last_player:
                                count += 1
                                if count == self.row_length:
                                    self.winner = last_player
                                    return
                            else:
                                break
                        else:
                            break
                    max_count = max(max_count, count)
        self.longest_lines[last_player - 1] = max_count
        # Check if the board is full
        if self.n_moves == pow(self.size, 2):
            self.winner = -1
            return

        self.winner = 0

    @cython.cfunc
    def is_capture(self, move: cython.tuple) -> cython.bint:
        # There are no captures in Tic-Tac-Toe, so this function always returns False.
        return False

    @cython.cfunc
    def evaluate_moves(self, moves: cython.list) -> cython.list:
        """
        :param moves: The list of moves to evaluate.
        :return: A list of tuples of moves and scores for each move.
        """
        scores: list[tuple] = [()] * len(moves)
        for i in range(len(moves)):
            scores[i] = (
                moves[i],
                self.evaluate_move(moves[i]),
            )
        return scores

    @cython.cfunc
    def move_weights(self, moves: cython.list) -> cython.list:
        """
        :param moves: The list of moves to evaluate.
        :return: A list of scores for each move.
        """
        scores = [0] * len(moves)
        for i in range(len(moves)):
            scores[i] = self.evaluate_move(moves[i])
        return scores

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    @cython.locals(
        x=cython.short,
        y=cython.short,
        conn_sc=cython.short,
        i=cython.short,
        j=cython.short,
        n_x=cython.short,
        n_y=cython.short,
    )
    def evaluate_move(self, move: cython.tuple) -> cython.int:
        """
        Evaluates the "connectivity" and "centrality" of a move.

        :param move: The move to evaluate.
        :return: A tuple with the freedom score, connectivity score, and centrality score for the move.
        """
        x: cython.int = move[0]
        y: cython.int = move[1]
        # Initialize the score
        conn_sc = 0
        # Iterate over all 8 possible adjacent positions
        for i in range(-1, 2):
            for j in range(-1, 2):
                # Exclude the current position itself
                if i != 0 or j != 0:
                    # Calculate the new potential position
                    n_x = x + i
                    n_y = y + j
                    # If the new position is within the board and is occupied by the current player, increment the connectivity_score
                    if 0 <= n_x < self.size and 0 <= n_y < self.size and self.board[n_x, n_y] == self.player:
                        conn_sc += 2
        return 10 + conn_sc + (self.center - abs(x - self.center)) + (self.center - abs(y - self.center))

    @cython.ccall
    @cython.locals(i=cython.short, j=cython.short)
    def visualize(self, full_debug=False):
        MARKS = {
            0: colored(" ", "white"),
            1: colored("X", "green", attrs=["bold"]),
            2: colored("O", "red", attrs=["bold"]),
        }

        visual = "  "
        # Column headers
        for j in range(self.size):
            visual += chr(ord("A") + j) + " "  # Using ASCII values to convert numbers to letters
        visual += "\n"  # Add a newline at the end

        for i in range(self.size):
            visual += chr(ord("A") + i) + " "  # Print row letters
            for j in range(self.size):
                visual += MARKS[self.board[i, j]] + " "
            visual += "\n"  # Newline at the end of each row

        visual += f"\nPlayer {self.player} | {self.n_moves} moves made | pie_move_done: {self.pie_move_done} | "
        visual += f"Winner: {self.winner if self.winner != 0 else 'None'}\n"
        visual += f"Longest lines: {self.longest_lines[0]} / {self.longest_lines[1]}\n"
        if full_debug:
            actions = self.get_legal_actions()
            visual += "hash: " + str(self.board_hash)
            visual += f"\nReward: ({self.get_reward(1)}/{self.get_reward(2)}), Terminal: {self.is_terminal()}"
            visual += f"\n# moves: {len(actions)} last_action: {actions_to_letters(self.last_action)}"

            visual += f"\nEv P1: {self.evaluate(1, params=self.default_params, norm=False)} / {self.evaluate(2, params=self.default_params, norm=False)} | "
            visual += f"normalized P1: {self.evaluate(1, params=self.default_params, norm=True)} / {self.evaluate(2, params=self.default_params, norm=True)}"
            # if len(actions) > 0:
            #     actions = self.evaluate_moves(self.get_legal_actions())
            #     actions = sorted(actions, key=lambda x: x[1], reverse=True)
            #     visual += "\n" + "..." * 60
            #     visual += "\n" + str(actions)
        return visual

    @property
    def transposition_table_size(self):
        # return an appropriate size based on the game characteristics
        return 2**18

    def __repr__(self) -> str:
        game: str = (
            "tictactoe " + str(self.size)
            if self.row_length == self.size
            else str(self.row_length) + "ninarow" + str(self.size)
        )
        return game

    param_order: dict = {"m_power": 0, "m_centre_bonus": 1, "a": 2}

    default_params = array.array("f", [2, 2, 40])

    @cython.cfunc
    @cython.exceptval(-9999999, check=False)
    @cython.locals(
        score_p1=cython.float,
        score_p2=cython.float,
        x=cython.char,
        y=cython.char,
        dx=cython.short,
        dy=cython.short,
        nx=cython.short,
        ny=cython.short,
        p=cython.short,
        o=cython.short,
        offset=cython.short,
        posi=cython.short,
        direct=cython.short,
        count=cython.short,
        space_count=cython.short,
        centrality_score=cython.double,
        positions=cython.list[cython.char],
        direction=cython.short,
        line_broken=cython.short,
        parts=cython.short,
    )
    def evaluate(
        self,
        player: cython.short,
        params: cython.float[:],
        norm: cython.bint = 0,
    ) -> cython.float:
        # "m_power": 0, "m_centre_bonus": 1, "a": 2
        score_p1 = 0
        score_p2 = 0

        positions = [-1] * 2
        for p in range(1, 3):
            positions[p - 1] = where_is_k2d(self.board, p)

        if self.n_moves < 2 * self.row_length:
            for p in range(1, 3):
                for posi in range(len(positions[p - 1])):
                    x = positions[p - 1][posi][0]
                    y = positions[p - 1][posi][1]
                    centrality_score = (self.center - abs(x - self.center)) + (self.center - abs(y - self.center))
                    if p == 1:
                        score_p1 += params[1] * centrality_score
                    else:
                        score_p2 += params[1] * centrality_score

        for p in range(1, 3):
            seen: cython.set = set()
            o = 3 - p
            for posi in range(len(positions[p - 1])):
                x = positions[p - 1][posi][0]
                y = positions[p - 1][posi][1]

                for dx in range(-1, 2):
                    for dy in range(0, 2):
                        if (dx != 0 or dy != 0) and not (dx == -1 and dy == 0):
                            # seen this position already in this direction, so skip it
                            if (x, y, dx, dy) in seen:
                                continue
                            # Counting marks, looking only in the positive direction
                            count = 1  # Start with the current mark
                            line_broken = 0
                            parts = 0
                            for offset in range(1, self.row_length):
                                nx = x + (dx * offset)
                                ny = y + (dy * offset)
                                if (
                                    nx < 0
                                    or nx >= self.board.shape[0]
                                    or ny < 0
                                    or ny >= self.board.shape[1]
                                    or self.board[nx, ny] == o
                                ):
                                    break
                                if self.board[nx, ny] == 0 and line_broken == 1:
                                    break
                                if self.board[nx, ny] == p and line_broken <= 1:
                                    count += 1
                                    # Looking at an unbroken line, we don't want to check this position again
                                    if line_broken == 0:
                                        seen.add((nx, ny, dx, dy))
                                    else:
                                        parts += 1

                                if self.board[nx, ny] == 0:
                                    line_broken += 1

                            if count <= 2:
                                continue

                            # Counting spaces, looking in both directions
                            space_count = count
                            for offset in range(1, self.row_length):
                                nx = x + (dx * offset)
                                ny = y + (dy * offset)
                                if (
                                    nx < 0
                                    or nx >= self.board.shape[0]
                                    or ny < 0
                                    or ny >= self.board.shape[1]
                                    or self.board[nx, ny] == o
                                ):
                                    break
                                if self.board[nx, ny] == 0 or self.board[nx, ny] == p:
                                    space_count += 1
                                    if space_count >= self.row_length:
                                        break

                            if space_count < self.row_length:
                                for offset in range(-1, -self.row_length, -1):
                                    nx = x + (dx * offset)
                                    ny = y + (dy * offset)
                                    if (
                                        nx < 0
                                        or nx >= self.board.shape[0]
                                        or ny < 0
                                        or ny >= self.board.shape[1]
                                        or self.board[nx, ny] == o
                                    ):
                                        break
                                    if self.board[nx, ny] == 0 or self.board[nx, ny] == p:
                                        space_count += 1
                                        if space_count >= self.row_length:
                                            break

                            # If the line can be potentially completed, add the count to the score
                            if space_count >= self.row_length:
                                # Prefer unbroken lines, unless we only need one more mark to win
                                if count != self.row_length - 1:
                                    power: cython.float = max(
                                        1, params[0] - parts
                                    )  # * Parts will be 0 if the line is unbroken
                                else:
                                    power: cython.float = params[0]

                                if p == 1:
                                    score_p1 += pow(count, power) + 0.2 * (space_count - count)
                                else:
                                    score_p2 += pow(count, power) + 0.2 * (space_count - count)

        if norm:
            if self.n_moves < 3 * self.row_length:
                return normalize(score_p1 - score_p2 if player == 1 else score_p2 - score_p1, params[2] / 2.0)
            else:
                return normalize(score_p1 - score_p2 if player == 1 else score_p2 - score_p1, params[2])

        return int(score_p1 - score_p2 if player == 1 else score_p2 - score_p1)


@cython.ccall
def actions_to_letters(action_tuple):
    """
    Converts a tuple of numerical actions into their corresponding letters.

    Parameters:
        action_tuple (tuple): A tuple containing numerical indices of the actions.

    Returns:
        tuple: A tuple containing the corresponding letters of the alphabet for each action index.
    """
    letters = []
    for action_number in action_tuple:
        if action_number < 0 or action_number >= 26:
            return action_tuple
        letter = chr(ord("A") + action_number)
        letters.append(letter)
    return tuple(letters)


def evaluate_tictactoe(self: GameState, player: int, m_opp_disc: float = 1.0, m_score: float = 1.0) -> float:
    """
    Evaluate the current self of a TicTacToe game.

    This function calculates a score for the current game self,
    favoring states where the specified player has more chances to win.

    Args:
        self: The current game self.
        player: The player for whom to evaluate the game self.
        score_discount (float): The discount to apply to the score if it is the opponent's turn.
        win_bonus (int): The bonus to add to the score if the game is potentially winnable on the next move.
        opp_disc: The ID of the opponent player.

    Returns:
        int: The score of the current game self for the specified player.
    """

    def calculate_score(marks, score_factor):
        if opponent not in marks and player in marks:
            return score_factor * ((size - marks.count(0)) * marks.count(player)) ** 2
        elif player not in marks and opponent in marks:
            return -score_factor * ((size - marks.count(0)) * marks.count(opponent)) ** 2
        return 0

    def potential_win(marks):
        return marks.count(player if self.player == player else opponent) == size - 1 and marks.count(0) == 1

    score = 0
    board = self.board
    size = self.size
    opponent = 3 - player

    lines = []

    # Rows and columns
    for i in range(size):
        lines.append([board[i][j] for j in range(size)])  # row i
        lines.append([board[j][i] for j in range(size)])  # column i

    # Diagonals
    lines.append([board[i][i] for i in range(size)])  # main diagonal
    lines.append([board[i][size - i - 1] for i in range(size)])  # anti-diagonal

    for marks in lines:
        score += calculate_score(marks, m_score)
        if potential_win(marks):
            return 10000 if self.player == player else -10000

    # Discount score if it is the opponent's turn
    score *= m_opp_disc if self.player == opponent else 1.0

    return score


cache: cython.dict = {}


def generate_masks(length: int, player: int, e=3) -> list:
    masks = []
    # Generate all possible masks with one missing entry
    for num_marks in range(3, length + 1):
        for mask_indices in itertools.combinations(range(length), num_marks):
            if num_marks == length:
                continue
            new_mask = np.zeros(length)
            new_mask[list(mask_indices)] = player

            # Special case: Generate the special mask (0 1 1 1 1 0)
            if num_marks == length - 1:
                special_mask = np.zeros(length + 1)
                special_mask[1:length] = player
                masks.append((special_mask, length**e))

            # Find connected segments by splitting the mask at zeros
            mask_str = "".join(map(str, new_mask.astype(int)))
            connected_segments = [len(segment) for segment in mask_str.split("0") if str(player) in segment]
            penalty = len(connected_segments) if num_marks < length - 1 else 0
            score = num_marks**e - (penalty - 1) * 2 * e

            masks.append(
                (
                    new_mask.reshape(
                        length,
                    ),
                    score,
                )
            )

    # Sort the masks by score in descending order
    masks.sort(key=lambda x: x[1], reverse=True)
    max_score = max([score for _, score in masks])
    # min_score = min([score for _, score in masks])
    # Normalize scores to range from 0 to 100
    for i in range(len(masks)):
        mask, score = masks[i]
        normalized_score = score / max_score * 100
        masks[i] = (mask, normalized_score)
    return masks


def masks_to_dict(masks):
    masks_dict = {}
    for mask, score in masks:
        # Convert the numpy array to a string and use it as a key for the dictionary
        key = "".join(map(str, mask.astype(int)))
        masks_dict[key] = score
    return masks_dict


@cython.ccall
@cython.infer_types(True)
def calculate_weights(m_top_k: cython.short, factor: cython.float, scale: cython.short = 10):
    """Calculates weights based on m_top_k and a configurable factor, using integers for intermediate calculations."""
    # Calculate weights proportional to their rank
    weights = [factor]
    remaining_weight = 1 - factor
    factor_scaled = int(factor * scale)

    for i in range(1, m_top_k):
        next_weight = remaining_weight * (factor_scaled - i) / (factor_scaled)
        weights.append(next_weight)
        remaining_weight -= next_weight

    # Due to potential floating-point precision issues, adjust the last weight
    weights[len(weights) - 1] = max(0, 1 - sum(weights[: len(weights) - 1]))

    return tuple(weights)


@cython.ccall
@cython.infer_types(True)
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
# Genetic optimization results:
# https://docs.google.com/spreadsheets/d/1cokehgvyvb5yIcfiAjI1v16h63czmYsOMF5dFiCLLn8/edit#gid=0
def evaluate_ninarow(
    state: cython.object,
    player: cython.short,
    m_bonus: cython.float = 9,
    m_decay: cython.float = 1.1,
    m_w_factor: cython.float = 0.95,
    m_top_k: cython.short = 2,
    m_disc: cython.float = 1.1,
    m_pow: cython.float = 7,
    norm: cython.bint = 0,
    a: cython.short = 100,
) -> cython.float:
    # Create a unique key based on the function parameters and the player
    key = (m_w_factor, m_top_k, m_pow)
    turns: cython.short = state.n_turns
    row_length: cython.short = state.row_length
    size: cython.short = state.size
    board: cnp.ndarray = state.board

    global cache
    # If the results are not already cached, compute them
    if key not in cache:
        cache[key] = {
            "player_masks": {
                1: masks_to_dict(generate_masks(state.row_length, 1, e=m_pow)),
                2: masks_to_dict(generate_masks(state.row_length, 2, e=m_pow)),
            },
            "m_weights": calculate_weights(m_top_k, m_w_factor),
        }

    # Use the cached results
    player_masks: cython.dict = cache[key]["player_masks"]
    m_weights: cython.tuple = cache[key]["m_weights"]

    # Extract the lines in each direction: rows, columns, and diagonals
    # line_set: cython.list = [None for _ in range(3)]

    # line_set[0] = board
    # line_set[1] = board.T

    # line_set[2] = [
    #     diag for d in range(-size + 1, size) for diag in (board.diagonal(d), np.fliplr(board).diagonal(d))
    # ]

    i: cython.short
    j: cython.short
    # Create a list to store all the lines
    all_lines: cython.list = []
    # Add all rows to the list
    for i in range(size):
        all_lines.append(board[i])
    trans_board: cnp.ndarray = np.transpose(board)
    # Add all columns to the list
    for i in range(size):
        all_lines.append(trans_board[i])
    flip_board: cnp.ndarray = np.fliplr(board)
    # Add all valid diagonals to the list
    for d in range(-size + 1, size):
        diag: cnp.ndarray = board.diagonal(d)
        if len(diag) >= row_length:
            all_lines.append(diag)
            all_lines.append(flip_board.diagonal(d))

    player_scores: cython.dict = {1: [], 2: []}
    decay: cython.float = 1.0 / (1.0 + (m_decay * float(turns)))

    if turns > ((row_length - 2) * 2) - 1:
        # Process rows
        for row in range(size):
            line = board[row, :]
            process_line(line, player_masks, player_scores, row_length)

        # Process columns
        trans_board = np.transpose(board)
        for col in range(size):
            line = trans_board[col, :]
            process_line(line, player_masks, player_scores, row_length)

        # Process diagonals
        flip_board = np.fliplr(board)
        for d in range(-size + 1, size):
            diag = board.diagonal(d)
            flip_diag = flip_board.diagonal(d)

            if len(diag) >= row_length:
                process_line(diag, player_masks, player_scores, row_length)
                process_line(flip_diag, player_masks, player_scores, row_length)

    score_bonus: cython.dict = {1: 0, 2: 0}
    if decay > 0.1:
        center: cython.short = size // 2
        # Returns a tuple of arrays, one for each dimension
        non_zero_indices: np.ndarray = np.nonzero(board)
        x: cython.short
        y: cython.short
        x_: cython.short
        y_: cython.short
        length: cython.Py_ssize_t = len(non_zero_indices[0])
        move: tuple[cython.short, cython.short]
        element: cython.short

        for k in range(length):
            x = non_zero_indices[0][k]
            y = non_zero_indices[1][k]
            # Go over the elements of the board
            element = board[x, y]
            # Add a bonus for player's marks close to the center
            score_bonus[element] += (center - abs(x - center)) + (center - abs(y - center)) / center
            # Add a bonus for player's marks close to other same player's marks

            adjacent_moves: cython.list = [(x + i, y + j) for i in [-1, 0, 1] for j in [-1, 0, 1] if i != 0 or j != 0]

            for j in range(len(adjacent_moves)):
                move = adjacent_moves[j]
                x_ = move[0]
                y_ = move[1]

                if 0 <= x_ < state.size and 0 <= y_ < state.size:
                    if board[x_, y_] == element:
                        score_bonus[element] += 1  # Note that the connection will be counted twice

    # Sort and take the top k scores, weigh them
    for p in player_scores:
        player_scores[p].sort(reverse=True)
        # player_scores[p] = sum(s * w for s, w in zip(player_scores[p][:m_top_k], m_weights)) # Python
        temp_sum = 0
        scores = player_scores[p][:m_top_k]
        for i in range(len(scores)):
            temp_sum += scores[i] * m_weights[i]
        player_scores[p] = temp_sum

    p1_score: cython.float = player_scores[1] + (decay * m_bonus * score_bonus[1])
    p2_score: cython.float = player_scores[2] + (decay * m_bonus * score_bonus[2])
    # The score is player 1's score minus player 2's score from the perspective of the provided player
    score: cython.float = (p1_score - p2_score) if player == 1 else (p2_score - p1_score)

    if m_disc > 0 and player != state.player:
        score *= m_disc

    if norm:
        return normalize(score, a)
    return score


@cython.ccall
@cython.infer_types(True)
def process_line(line, player_masks, player_scores, row_length):
    for x in range(line.shape[0] - row_length + 1):
        line_segment_str = ""
        for i in range(x, x + row_length):
            line_segment_str += str(int(line[i]))

        for p in range(1, 3):
            max_mask_player = max(player_masks[p].get(line_segment_str, 0), 0)
            player_scores[p].append(max_mask_player)

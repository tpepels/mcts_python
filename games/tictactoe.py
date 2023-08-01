# cython: language_level=3, initializedcheck=False

import itertools
import random
import cython
import numpy as np

from cython.cimports import numpy as cnp

cnp.import_array()
import numpy as np
from games.gamestate import GameState, win, loss, draw, normalize
from termcolor import colored

# MARKS = {0: " ", 1: "X", 2: "O"}

if cython.compiled:
    print("Tictactoe is compiled.")
else:
    print("Tictactoe is just a lowly interpreted script.")


class TicTacToeGameState(GameState):
    zobrist_tables = {
        size: [[[random.randint(1, 2**32 - 1) for _ in range(3)] for _ in range(size)] for _ in range(size)]
        for size in range(3, 10)
    }

    def __init__(
        self, board_size=3, row_length=None, last_move=None, board=None, player=1, n_turns=0, board_hash=None
    ):
        self.size = board_size
        self.row_length = row_length if row_length else board_size
        self.board: np.ndarray = board
        self.board_hash = board_hash
        self.player = player

        self.last_move = last_move
        self.n_turns = n_turns

        self.zobrist_table = self.zobrist_tables[self.size]

        if self.board is None:
            self.board = np.zeros((self.size, self.size), dtype=np.int32)
            self.board_hash = 0
            for i in range(self.size):
                for j in range(self.size):
                    piece = self.board[i][j]
                    self.board_hash ^= self.zobrist_table[i][j][piece]

    def apply_action(self, action):
        x, y = action
        if self.board[x, y] != 0:
            raise ValueError("Illegal move")

        new_board = np.copy(self.board)
        new_board[x][y] = self.player
        board_hash = self.board_hash ^ self.zobrist_table[x][y][0] ^ self.zobrist_table[x][y][3 - self.player]

        new_state = TicTacToeGameState(
            board_size=self.size,
            board=new_board,
            player=3 - self.player,
            row_length=self.row_length,
            board_hash=board_hash,
            n_turns=self.n_turns + 1,
            last_move=action,
        )
        return new_state

    def skip_turn(self):
        """Used for the null-move heuristic in alpha-beta search"""
        new_board = np.copy(self.board)
        # Pass the same hash since this is only used for null-moves
        return TicTacToeGameState(
            board_size=self.size,
            board=new_board,
            player=3 - self.player,
            row_length=self.row_length,
            board_hash=self.board_hash,
            n_turns=self.n_turns,
            last_move=None,
        )

    def get_random_action(self):
        return next(self.yield_legal_actions(), None)

    def yield_legal_actions(self):
        all_indices = list(itertools.product(range(self.size), range(self.size)))
        random.shuffle(all_indices)  # Shuffle the list
        for i, j in all_indices:
            if self.board[i, j] == 0:
                yield (i, j)

    def get_legal_actions(self):
        return list(zip(*np.where(self.board == 0)))

    def is_terminal(self):
        if self.n_turns >= (self.row_length * 2) - 1:
            return (self.n_turns == self.size**2) or self.get_reward(1) != 0
        return False

    def get_reward(self, player):
        # We first need enough marks on the board to be able to win
        if self.n_turns < (self.row_length * 2) - 1:
            return 0

        return get_reward(
            player, self.last_move[0], self.last_move[1], self.board, self.row_length, self.size, self.player
        )

    def is_capture(self, move):
        # There are no captures in Tic-Tac-Toe, so this function always returns False.
        return False

    def evaluate_moves(self, moves):
        """
        :param moves: The list of moves to evaluate.
        :return: A list of tuples of moves and scores for each move.
        """
        scores: list[tuple] = [()] * len(moves)
        for i in range(len(moves)):
            scores[i] = (
                moves[i],
                evaluate_move(moves[i][0], moves[i][1], self.size, self.board, self.player),
            )
        return scores

    def move_weights(self, moves):
        """
        :param moves: The list of moves to evaluate.
        :return: A list of scores for each move.
        """
        scores = [0] * len(moves)
        for i in range(len(moves)):
            scores[i] = evaluate_move(moves[i][0], moves[i][1], self.size, self.board, self.player)
        return scores

    def evaluate_move(self, move):
        """
        Evaluates the "connectivity" and "centrality" of a move.

        :param move: The move to evaluate.
        :return: A tuple with the freedom score, connectivity score, and centrality score for the move.
        """
        return evaluate_move(move[0], move[1], self.size, self.board, self.player)

    def visualize(self, full_debug=False):
        MARKS = {
            0: colored(" ", "white"),
            1: colored("X", "green", attrs=["bold"]),
            2: colored("O", "red", attrs=["bold"]),
        }

        visual = "  " + "  ".join(str(i) for i in range(self.size)) + "\n"  # Print column numbers

        for i in range(self.size):
            visual += str(i) + " "  # Print row numbers
            for j in range(self.size):
                visual += MARKS[self.board[i, j]] + "  "  # Adding two spaces
            visual = visual[:-2] + "\n"  # Remove extra spaces at the end of the line and reset color

        if full_debug:
            actions = self.get_legal_actions()
            visual += "hash: " + str(self.board_hash)
            visual += f"\nPlayer: {self.player} | {self.n_turns} moves made"
            visual += f"\nReward: ({self.get_reward(1)}/{self.get_reward(2)}), Terminal: {self.is_terminal()}"
            visual += f"\n{len(actions)} last_move: {self.last_move} actions: {self.get_legal_actions()}"

            visual += f"\nEv P1: {evaluate_ninarow(self, 1)}"
            visual += f"\nEv P2: {evaluate_ninarow(self, 2)}"
            visual += f"\nSimple P1: {evaluate_ninarow_fast(self, 1)}"
            # visual += f"\nSimple P2: {ninarow_simple_evaluation(self, 2)}"
            if len(actions) > 0:
                actions = self.evaluate_moves(self.get_legal_actions())
                actions = sorted(actions, key=lambda x: x[1], reverse=True)
                visual += "\n" + "..." * 60
                visual += "\n" + str(actions)
        return visual

    @property
    def transposition_table_size(self):
        # return an appropriate size based on the game characteristics
        return 2 ** (self.size * 2 + 1)

    def __repr__(self) -> str:
        game: str = (
            "tictactoe " + str(self.size)
            if self.row_length == self.size
            else str(self.row_length) + "ninarow" + str(self.size)
        )
        return game


@cython.ccall
@cython.infer_types(True)
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.locals(
    x=cython.int,
    y=cython.int,
    center=cython.int,
    size=cython.int,
    connectivity_score=cython.int,
    centrality_score=cython.int,
    i=cython.int,
    j=cython.int,
    new_x=cython.int,
    new_y=cython.int,
    board=cython.int[:, :],
    player=cython.int,
)
def evaluate_move(x, y, size, board, player) -> cython.int:
    # Initialize the score
    connectivity_score = 0

    # Calculate the Manhattan distance from the center
    center = size // 2

    # Iterate over all 8 possible adjacent positions
    for i in range(-1, 2):
        for j in range(-1, 2):
            # Exclude the current position itself
            if i != 0 or j != 0:
                # Calculate the new potential position
                new_x = x + i
                new_y = y + j

                # If the new position is within the board and is occupied by the current player, increment the connectivity_score
                if 0 <= new_x < size and 0 <= new_y < size and board[new_x, new_y] == player:
                    connectivity_score += 3

    # Calculate the centrality score
    centrality_score = (center - abs(x - center)) + (center - abs(y - center))

    return 10 + connectivity_score + centrality_score


directions: cython.list = [(0, 1), (1, 0), (1, 1), (-1, 1)]  # horizontal, vertical, two diagonal directions


@cython.ccall
@cython.infer_types(True)
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.locals(
    player=cython.int,
    last_move_x=cython.int,
    last_move_y=cython.int,
    board=cython.int[:, :],
    row_length=cython.int,
    size=cython.int,
    last_player=cython.int,
    am_i_last_player=cython.bint,
    dx=cython.int,
    dy=cython.int,
    count=cython.int,
    i=cython.int,
    x=cython.int,
    y=cython.int,
    player_to_move=cython.int,
)
def get_reward(player, last_move_x, last_move_y, board, row_length, size, player_to_move) -> cython.int:
    # only the last player to make a move can actually win the game, so we can look from their perspective
    last_player = 3 - player_to_move
    am_i_last_player = player == last_player

    # Check all 8 directions around the last move made
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            count = 1
            # Start from the last move and check for row_length cells

            for i in range(-(row_length + 1), row_length):
                if i == 0:  # Skip the starting cell
                    continue
                x = last_move_x + (dx * i)
                y = last_move_y + (dy * i)

                # Ensure the indices are within the board
                if 0 <= x < size and 0 <= y < size:
                    if board[x, y] == last_player:
                        count += 1
                        if count == row_length:
                            return win if am_i_last_player else loss
    return 0


def evaluate_tictactoe(state: GameState, player: int, m_opp_disc: float = 1.0, m_score: float = 1.0) -> float:
    """
    Evaluate the current state of a TicTacToe game.

    This function calculates a score for the current game state,
    favoring states where the specified player has more chances to win.

    Args:
        state: The current game state.
        player: The player for whom to evaluate the game state.
        score_discount (float): The discount to apply to the score if it is the opponent's turn.
        win_bonus (int): The bonus to add to the score if the game is potentially winnable on the next move.
        opp_disc: The ID of the opponent player.

    Returns:
        int: The score of the current game state for the specified player.
    """

    def calculate_score(marks, score_factor):
        if opponent not in marks and player in marks:
            return score_factor * ((size - marks.count(0)) * marks.count(player)) ** 2
        elif player not in marks and opponent in marks:
            return -score_factor * ((size - marks.count(0)) * marks.count(opponent)) ** 2
        return 0

    def potential_win(marks):
        return marks.count(player if state.player == player else opponent) == size - 1 and marks.count(0) == 1

    score = 0
    board = state.board
    size = state.size
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
            return 10000 if state.player == player else -10000

    # Discount score if it is the opponent's turn
    score *= m_opp_disc if state.player == opponent else 1.0

    return score


directions: cython.list = [(1, 0), (0, 1), (1, 1), (-1, 1)]  # right, down, down-right, up-right


@cython.ccall
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.locals(
    row_length=cython.int,
    board=cython.int[:, :],
    score_p1=cython.double,
    score_p2=cython.double,
    x=cython.int,
    y=cython.int,
    dx=cython.int,
    dy=cython.int,
    nx=cython.int,
    ny=cython.int,
    p=cython.int,
    o=cython.int,
    offset=cython.int,
    posi=cython.int,
    direct=cython.int,
    count=cython.int,
    space_count=cython.int,
    center=cython.int,
    centrality_score=cython.double,
    positions=cython.int[:, :],
    direction=cython.int,
    line_broken=cython.int,
    parts=cython.int,
)
def evaluate_ninarow_fast(
    state: cython.object,
    player: cython.int,
    m_power: cython.int = 2,
    m_opp_disc: cython.double = 1.0,
    m_centre_bonus: cython.double = 1.0,
    norm: cython.bint = 0,
    a: cython.int = 100,
) -> cython.int:
    row_length = state.row_length
    board = state.board
    center = board.shape[0] // 2
    score_p1 = 0
    score_p2 = 0

    if state.n_turns < 3 * row_length:
        for p in range(1, 3):
            positions = np.array(np.where(state.board == p)).T.astype(np.int32)
            for posi in range(positions.shape[0]):
                x = positions[posi][0]
                y = positions[posi][1]
                centrality_score = (center - abs(x - center)) + (center - abs(y - center))
                if p == 1:
                    score_p1 += m_centre_bonus * centrality_score
                else:
                    score_p2 += m_centre_bonus * centrality_score

    if state.n_turns > row_length:
        for p in range(1, 3):
            seen: cython.set = set()
            o = 3 - p
            positions = np.array(np.where(state.board == p)).T.astype(np.int32)
            for posi in range(positions.shape[0]):
                x = positions[posi][0]
                y = positions[posi][1]

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
                            for offset in range(1, row_length):
                                nx = x + (dx * offset)
                                ny = y + (dy * offset)
                                if (
                                    nx < 0
                                    or nx >= board.shape[0]
                                    or ny < 0
                                    or ny >= board.shape[1]
                                    or board[nx, ny] == o
                                ):
                                    break
                                if board[nx, ny] == 0 and line_broken == 1:
                                    break
                                if board[nx, ny] == p and line_broken <= 1:
                                    count += 1
                                    # Looking at an unbroken line, we don't want to check this position again
                                    if line_broken == 0:
                                        seen.add((nx, ny, dx, dy))
                                    else:
                                        parts += 1
                                if board[nx, ny] == 0:
                                    line_broken += 1

                            if count < 2:
                                continue

                            # Counting spaces, looking in both directions
                            space_count = 1
                            for offset in range(1, row_length):
                                nx = x + (dx * offset)
                                ny = y + (dy * offset)
                                if (
                                    nx < 0
                                    or nx >= board.shape[0]
                                    or ny < 0
                                    or ny >= board.shape[1]
                                    or board[nx, ny] == o
                                ):
                                    break
                                if board[nx, ny] == 0 or board[nx, ny] == p:
                                    space_count += 1
                                    if space_count >= row_length:
                                        break

                            if space_count < row_length:
                                for offset in range(-1, -row_length, -1):
                                    nx = x + (dx * offset)
                                    ny = y + (dy * offset)
                                    if (
                                        nx < 0
                                        or nx >= board.shape[0]
                                        or ny < 0
                                        or ny >= board.shape[1]
                                        or board[nx, ny] == o
                                    ):
                                        break
                                    if board[nx, ny] == 0 or board[nx, ny] == p:
                                        space_count += 1
                                        if space_count >= row_length:
                                            break

                            # If the line can be potentially completed, add the count to the score
                            if space_count >= row_length:
                                # print(
                                #     f"Player {p} has {count} marks in a row at ({x}, {y}) with {space_count} spaces line_broken={line_broken}, parts={parts}"
                                # )
                                # If the line is broken we don't want that, unless we only need one more mark to win
                                if count < (row_length - 1):
                                    count -= parts

                                if p == 1:
                                    score_p1 += count**m_power
                                else:
                                    score_p2 += count**m_power

                            # else:
                            #     print(
                            #         f"Player {p} has {count} marks in a row at ({x}, {y}) but the line is not long enough with {space_count} spaces line_broken={line_broken}, parts={parts}"
                            #     )
    if norm:
        return normalize(score_p1 - score_p2 if player == 1 else score_p2 - score_p1, a)

    if state.player == player:
        return int(score_p1 - score_p2 if player == 1 else score_p2 - score_p1)
    else:
        return int(score_p1 - score_p2 if player == 1 else score_p2 - score_p1) * m_opp_disc


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
def calculate_weights(m_top_k: cython.int, factor: cython.float, scale: cython.int = 10):
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
    weights[-1] = max(0, 1 - sum(weights[:-1]))

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
    player: cython.int,
    m_bonus: cython.float = 18,
    m_decay: cython.float = 3.5,
    m_w_factor: cython.float = 0.6,
    m_top_k: cython.int = 4,
    m_disc: cython.float = 0.7,
    m_pow: cython.float = 9,
    norm: cython.bint = 0,
    a: cython.int = 100,
) -> cython.float:
    # Create a unique key based on the function parameters and the player
    key = (m_w_factor, m_top_k, m_pow)
    turns: cython.int = state.n_turns
    row_length: cython.int = state.row_length
    size: cython.int = state.size
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

    i: cython.int
    j: cython.int
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
        center: cython.int = size // 2
        # Returns a tuple of arrays, one for each dimension
        non_zero_indices: np.ndarray = np.nonzero(board)
        x: cython.int
        y: cython.int
        x_: cython.int
        y_: cython.int
        length: cython.int = len(non_zero_indices[0])
        move: tuple[cython.int, cython.int]
        element: cython.int

        for k in range(length):
            x = non_zero_indices[0][k]
            y = non_zero_indices[1][k]
            # Go over the elements of the board
            element = board[x, y]
            # Add a bonus for player's marks close to the center
            score_bonus[element] += (center - abs(x - center)) + (center - abs(y - center)) / center
            # Add a bonus for player's marks close to other same player's marks

            adjacent_moves: cython.list = [
                (x + i, y + j) for i in [-1, 0, 1] for j in [-1, 0, 1] if i != 0 or j != 0
            ]

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

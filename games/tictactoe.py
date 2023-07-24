# cython: language_level=3
# cython: infer_types=True

import itertools
import random
import cython
import numpy as np

from cython.cimports import numpy as cnp

cnp.import_array()

import numpy as np
from games.gamestate import GameState, win, loss, draw, normalize

MARKS = {0: " ", 1: "X", 2: "O"}

if cython.compiled:
    print("Tictactoe is compiled.")
else:
    print("Tictactoe is just a lowly interpreted script.")


class TicTacToeGameState(GameState):
    players_bitstrings = [random.randint(1, 2**64 - 1) for _ in range(3)]  # 0 is for the empty player
    zobrist_tables = {
        size: [[[random.randint(1, 2**64 - 1) for _ in range(3)] for _ in range(size)] for _ in range(size)]
        for size in range(3, 10)
    }

    def __init__(
        self, board_size=3, row_length=None, last_move=None, board=None, player=1, n_turns=0, board_hash=None
    ):
        self.size = board_size
        self.row_length = row_length if row_length else self.size
        self.board = board
        self.board_hash = board_hash
        self.player = player

        self.last_move = last_move
        self.n_turns = n_turns

        self.zobrist_table = self.zobrist_tables[self.size]

        if self.board is None:
            self.board = np.zeros((self.size, self.size), dtype=int)
            self.board_hash = 0
            for i in range(self.size):
                for j in range(self.size):
                    piece = self.board[i][j]
                    self.board_hash ^= self.zobrist_table[i][j][piece]
            self.board_hash ^= self.players_bitstrings[self.player]

    def apply_action(self, action):
        x, y = action
        if self.board[x, y] != 0:
            raise ValueError("Illegal move")

        new_board = np.copy(self.board)
        new_board[x][y] = self.player
        board_hash = (
            self.board_hash
            ^ self.zobrist_table[x][y][0]
            ^ self.zobrist_table[x][y][3 - self.player]
            ^ self.players_bitstrings[self.player]
            ^ self.players_bitstrings[3 - self.player]
        )

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
        non_zero_indices = np.transpose(np.nonzero(self.board)).tolist()  # Convert to list of pairs
        random.shuffle(non_zero_indices)  # Shuffle the list

        for i, j in non_zero_indices:
            yield (i, j)

    def get_legal_actions(self):
        return list(zip(*np.where(self.board == 0)))

    def is_terminal(self):
        if self.n_turns > self.row_length * 2 - 1:
            return (np.count_nonzero(self.board == 0) == 0) or (self.get_reward(1) != 0)

    def get_reward(self, player):
        assert self.last_move is not None

        # We first need enough marks on the board to be able to win
        if self.n_turns < self.row_length * 2:
            return 0

        return get_reward(player, self.last_move, self.board, self.row_length, self.size, self.player)

    def is_capture(self, move):
        # There are no captures in Tic-Tac-Toe, so this function always returns False.
        return False

    def evaluate_moves(self, moves):
        """
        Evaluates the "connectivity" and "centrality" of a list of moves.

        :param moves: The list of moves to evaluate.
        :return: A list of scores for each move.
        """
        scores = []
        for move in moves:
            scores.append((move, evaluate_move(move, self.size, self.board, self.player)))
        return scores

    def evaluate_move(self, move):
        """
        Evaluates the "connectivity" and "centrality" of a move.

        :param move: The move to evaluate.
        :return: A tuple with the freedom score, connectivity score, and centrality score for the move.
        """
        return evaluate_move(move, self.size, self.board, self.player)

    def visualize(self):
        visual = "    " + "   ".join(str(i) for i in range(self.size)) + "\n"  # Print column numbers
        visual += "    " + "----" * self.size + "\n"

        for i in range(self.size):
            visual += str(i) + "   "  # Print row numbers
            for j in range(self.size):
                visual += MARKS[self.board[i, j]] + " | "
            visual = visual[:-3] + "\n"  # Remove last separator

            if i != self.size - 1:
                visual += "    " + "----" * self.size + "\n"

        visual += "hash: " + str(self.board_hash)
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
    connectivity_score=cython.float,
    centrality_score=cython.float,
    i=cython.int,
    j=cython.int,
    a=cython.int,
    move=cython.tuple,
    board=cnp.ndarray,
    player=cython.int,
)
def evaluate_move(move, size, board, player) -> cython.float:
    x = move[0]
    y = move[1]
    adjacent_moves = [(x + i, y + j) for i in [-1, 0, 1] for j in [-1, 0, 1] if i != 0 or j != 0]
    # Calculate the Manhattan distance from the center
    center = size // 2
    connectivity_score = 0
    for a in range(len(adjacent_moves)):
        i = adjacent_moves[a][0]
        j = adjacent_moves[a][1]
        if 0 <= i < size and 0 <= j < size and board[i, j] == player:
            connectivity_score += 1

    centrality_score = ((center - abs(x - center)) + (center - abs(y - center))) / center

    return (2.0 * connectivity_score) + centrality_score


directions: cython.list = [(0, 1), (1, 0), (1, 1), (-1, 1)]  # horizontal, vertical, two diagonal directions


@cython.ccall
@cython.infer_types(True)
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.locals(
    player=cython.int,
    last_move=cython.tuple,
    board=cnp.ndarray,
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
)
def get_reward(player, last_move, board, row_length, size, player_to_move) -> cython.int:
    # only the last player to make a move can actually win the game, so we can look from their perspective
    last_player = 3 - player_to_move
    am_i_last_player = player == last_player

    for dx, dy in directions:
        count = 0
        for i in range(-row_length + 1, row_length):
            x = last_move[0] + dx * i
            y = last_move[1] + dy * i

            # Ensure the indices are within the board
            if 0 <= x < size and 0 <= y < size:
                if board[x, y] == last_player:
                    count += 1
                    if count == row_length:
                        return win if am_i_last_player else loss
                else:
                    count = 0
            else:
                count = 0

    # Check if the game is a draw
    if np.all(board != 0):  # If there are no empty spaces left
        return draw

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
def ninarow_simple_evaluation(
    state: cython.object,
    player: cython.int,
    power: cython.int = 4,
    norm: cython.bint = 0,
    a: cython.int = 100,
) -> cython.int:
    row_length: cython.int = state.row_length
    board: cnp.ndarray = state.board
    score_p1: cython.int = 0
    score_p2: cython.int = 0
    x: cython.int
    y: cython.int
    dx: cython.int
    dy: cython.int
    nx: cython.int
    ny: cython.int
    p: cython.int
    o: cython.int
    offset: cython.int
    posi: cython.int
    direct: cython.int
    count: cython.int
    space_count: cython.int
    center: cython.int = board.shape[0] // 2
    centrality_score: cython.int
    positions: cnp.ndarray
    direction: cython.int

    if state.n_turns < 3 * row_length:
        for p in range(1, 3):
            positions = np.array(np.where(board == p)).T
            for posi in range(positions.shape[0]):
                x = positions[posi][0]
                y = positions[posi][1]
                centrality_score = (center - abs(x - center)) + (center - abs(y - center))
                if p == 1:
                    score_p1 += centrality_score
                else:
                    score_p2 += centrality_score

    if state.n_turns > row_length:
        for p in range(1, 3):
            o = 3 - p
            positions = np.array(np.where(board == p)).T
            for posi in range(positions.shape[0]):
                x = positions[posi][0]
                y = positions[posi][1]

                for direct in range(4):
                    dx = directions[direct][0]
                    dy = directions[direct][1]

                    # Check if there's enough space in both directions
                    space_count = 0
                    for direction in (-1, 1):
                        for offset in range(1, row_length):
                            nx = x + (direction * dx * offset)
                            ny = y + (direction * dy * offset)
                            if (
                                nx < 0
                                or nx >= board.shape[0]
                                or ny < 0
                                or ny >= board.shape[1]
                                or board[nx, ny] == o
                            ):
                                break
                            space_count += 1

                    if space_count < row_length:
                        continue

                    count = 1  # Start with the current mark
                    for offset in range(1, row_length):
                        nx = x + dx * offset
                        ny = y + dy * offset

                        if (
                            nx < 0
                            or nx >= board.shape[0]
                            or ny < 0
                            or ny >= board.shape[1]
                            or board[nx, ny] != p
                        ):
                            break
                        count += 1

                    # If the line can be potentially completed, add the count to the score
                    if space_count >= row_length:
                        if p == 1:
                            score_p1 += int(count**power)
                        else:
                            score_p2 += int(count**power)

    if norm:
        return normalize(score_p1 - score_p2 if player == 1 else score_p2 - score_p1, a)
    return score_p1 - score_p2 if player == 1 else score_p2 - score_p1


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
def evaluate_n_in_a_row(
    state: cython.object,
    player: cython.int,
    m_bonus: cython.float = 0.75,
    m_decay: cython.float = 2.9,
    m_w_factor: cython.float = 0.8,
    m_top_k: cython.int = 4,
    m_disc: cython.float = 1.49,
    m_pow: cython.float = 7,
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
        move: cython.tuple
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

            for j in range(length):
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

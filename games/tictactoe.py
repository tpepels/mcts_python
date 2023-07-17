import itertools
import random

import numpy as np
from games.gamestate import GameState, win, loss, draw, normalize

MARKS = {0: " ", 1: "X", 2: "O"}


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
        if self.board[x][y] != 0:
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
        return random.choice(self.get_legal_actions())

    def yield_legal_actions(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    yield (i, j)

    def get_legal_actions(self):
        return list(zip(*np.where(self.board == 0)))

    def is_terminal(self):
        return np.count_nonzero(self.board == 0) == 0 or self.get_reward(1) != 0

    def get_reward(self, player):
        if self.last_move is None:
            return 0
        # We first need enough marks on the board to be able to win
        if self.n_turns < self.row_length * 2 - 1:
            return 0

        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]  # horizontal, vertical, two diagonal directions
        # only the last player to make a move can actually win the game, so we can look from their perspective
        last_player = 3 - self.player
        am_i_last_player = player == last_player

        for dx, dy in directions:
            count = 0
            for i in range(-self.row_length + 1, self.row_length):
                x = self.last_move[0] + dx * i
                y = self.last_move[1] + dy * i

                # Ensure the indices are within the board
                if 0 <= x < self.size and 0 <= y < self.size:
                    if self.board[x, y] == last_player:
                        count += 1
                        if count == self.row_length:
                            return win if am_i_last_player else loss
                    else:
                        count = 0
                else:
                    count = 0

        # Check if the game is a draw
        if np.all(self.board != 0):  # If there are no empty spaces left
            return draw

        return 0  # Otherwise, the game is not yet decided

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
            x, y = move
            adjacent_moves = [(x + i, y + j) for i in [-1, 0, 1] for j in [-1, 0, 1] if i != 0 or j != 0]

            connectivity_score = 0

            for i, j in adjacent_moves:
                if 0 <= i < self.size and 0 <= j < self.size:
                    if self.board[i][j] == self.player:
                        connectivity_score += 1

            # Calculate the Manhattan distance from the center
            center = self.size // 2
            centrality_score = -abs(x - center) - abs(y - center)

            scores.append((move, connectivity_score * self.size + centrality_score))
        return scores

    def evaluate_move(self, move):
        """
        Evaluates the "connectivity" and "centrality" of a move.

        :param move: The move to evaluate.
        :return: A tuple with the freedom score, connectivity score, and centrality score for the move.
        """
        x, y = move
        adjacent_moves = [(x + i, y + j) for i in [-1, 0, 1] for j in [-1, 0, 1] if i != 0 or j != 0]

        connectivity_score = 0

        for i, j in adjacent_moves:
            if 0 <= i < self.size and 0 <= j < self.size:
                if self.board[i][j] == self.player:
                    connectivity_score += 1

        # Calculate the Manhattan distance from the center
        center = self.size // 2
        centrality_score = ((center - abs(x - center)) + (center - abs(y - center))) / center

        return (2 * connectivity_score) + centrality_score

    def visualize(self):
        visual = "    " + "   ".join(str(i) for i in range(self.size)) + "\n"  # Print column numbers
        visual += "    " + "----" * self.size + "\n"

        for i in range(self.size):
            visual += str(i) + "   "  # Print row numbers
            for j in range(self.size):
                visual += MARKS[self.board[i][j]] + " | "
            visual = visual[:-3] + "\n"  # Remove last separator

            if i != self.size - 1:
                visual += "    " + "----" * self.size + "\n"

        visual += "hash: " + str(self.board_hash)
        return visual

    @property
    def transposition_table_size(self):
        # return an appropriate size based on the game characteristics
        return 2**16

    def __repr__(self) -> str:
        if self.row_length == self.size:
            return f"tic-tac-toe{self.size}"
        else:
            return f"{self.row_length}-in-a-row{self.size}"


def evaluate_tictactoe(state, player, m_opp_disc: float = 1.0, m_score: float = 1.0):
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


def generate_masks(length, player, e=3):
    print("generating masks for p" + str(player))
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


# def evaluate_n_in_a_row(state: TicTacToeGameState, player: int, norm: bool = False, a=100):
#     if not hasattr(evaluate_n_in_a_row, "player1_masks"):
#         masks_p1 = generate_masks(state.row_length, 1)
#         evaluate_n_in_a_row.player1_masks = masks_to_dict(masks_p1)

#         masks_p2 = generate_masks(state.row_length, 2)
#         evaluate_n_in_a_row.player2_masks = masks_to_dict(masks_p2)

#     # Extract the lines in each direction: rows, columns, and diagonals
#     rows = state.board
#     columns = state.board.T
#     diagonals = [
#         diag
#         for d in range(-state.board.shape[0] + 1, state.board.shape[1])
#         for diag in (state.board.diagonal(d), np.fliplr(state.board).diagonal(d))
#     ]

#     player1_scores = []
#     player2_scores = []

#     for line_set in [rows, columns, diagonals]:
#         for line in line_set:
#             # Ensure the line is long enough to contain the pattern
#             if len(line) < state.row_length:
#                 continue

#             max_mask_player1 = 0
#             max_mask_player2 = 0

#             for i in range(len(line) - state.row_length + 1):
#                 # Go over the segments of the line
#                 line_segment_str = "".join(str(int(e)) for e in line[i : i + state.row_length])

#                 # Look up the score of the segment in the mask dictionaries
#                 max_mask_player1 = max(
#                     evaluate_n_in_a_row.player1_masks.get(line_segment_str, 0), max_mask_player1
#                 )
#                 max_mask_player2 = max(
#                     evaluate_n_in_a_row.player2_masks.get(line_segment_str, 0), max_mask_player2
#                 )

#             player1_scores.append(max_mask_player1)
#             player2_scores.append(max_mask_player2)

#     # Sort and take the top 3 scores
#     player1_scores.sort(reverse=True)
#     player2_scores.sort(reverse=True)
#     player2_score = sum(player2_scores[:3])
#     player1_score = sum(player1_scores[:3])

#     # The score is player 1's score minus player 2's score from the perspective of the provided player
#     score = (player1_score - player2_score) if player == 1 else (player2_score - player1_score)
#     if norm:
#         return normalize(score, a)
#     return score


def evaluate_n_in_a_row(
    state: TicTacToeGameState,
    player: int,
    m_bonus=1,
    m_decay=0.95,
    m_weights=(0.7, 0.2, 0.1),
    m_top_k=3,
    m_disc=0.9,
    m_pow=4,
    norm: bool = False,
    a=100,
):
    if not hasattr(evaluate_n_in_a_row, "player_masks"):
        evaluate_n_in_a_row.player_masks = {
            1: masks_to_dict(generate_masks(state.row_length, 1, e=m_pow)),
            2: masks_to_dict(generate_masks(state.row_length, 2, e=m_pow)),
        }

    # Extract the lines in each direction: rows, columns, and diagonals
    rows = state.board
    columns = state.board.T
    diagonals = [
        diag
        for d in range(-state.board.shape[0] + 1, state.board.shape[1])
        for diag in (state.board.diagonal(d), np.fliplr(state.board).diagonal(d))
    ]

    player_scores = {1: [], 2: []}
    score_bonus = {1: 0, 2: 0}

    decay = 1 / (1 + (m_decay * state.n_turns))
    if state.n_turns > ((state.row_length - 2) * 2) - 1:
        for line_set in [rows, columns, diagonals]:
            for line in line_set:
                # Ensure the line is long enough to contain the pattern
                if len(line) < state.row_length:
                    continue
                for x in range(len(line) - state.row_length + 1):
                    # Go over the segments of the line
                    line_segment_str = "".join(str(int(e)) for e in line[x : x + state.row_length])

                    # Look up the score of the segment in the mask dictionaries for each player
                    for p in player_scores:
                        max_mask_player = max(evaluate_n_in_a_row.player_masks[p].get(line_segment_str, 0), 0)
                        player_scores[p].append(max_mask_player)

    if decay > 0.1:
        center = state.size // 2
        non_zero_indices = np.nonzero(state.board)  # Returns a tuple of arrays, one for each dimension

        for x, y in zip(*non_zero_indices):
            # Go over the elements of the board
            element = state.board[x, y]
            # Add a bonus for player's marks close to the center
            score_bonus[element] += (center - abs(x - center)) + (center - abs(y - center)) / center
            # Add a bonus for player's marks close to other same player's marks

            adjacent_moves = [(x + i, y + j) for i in [-1, 0, 1] for j in [-1, 0, 1] if i != 0 or j != 0]

            for i, j in adjacent_moves:
                if 0 <= i < state.size and 0 <= j < state.size:
                    if state.board[i, j] == element:
                        score_bonus[element] += 1  # Note that the connection will be counted twice

    try:
        # Sort and take the top k scores, weigh them
        for p in player_scores:
            player_scores[p].sort(reverse=True)
            player_scores[p] = sum(s * w for s, w in zip(player_scores[p][:m_top_k], m_weights))
    except TypeError as e:
        print(f"{m_bonus=}, {m_decay=}, {m_weights=}, {m_top_k=}, {m_disc=}, {m_pow=}")
        raise e
    final_scores = {p: player_scores[p] + (decay * m_bonus * score_bonus[p]) for p in player_scores}

    # The score is player 1's score minus player 2's score from the perspective of the provided player
    score = (final_scores[1] - final_scores[2]) if player == 1 else (final_scores[2] - final_scores[1])
    if m_disc > 0 and player != state.player:
        score *= m_disc

    if norm:
        return normalize(score, a)
    return score

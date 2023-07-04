import random
from games.gamestate import GameState, win, loss, draw, normalize

MARKS = {0: " ", 1: "X", 2: "O"}


class TicTacToeGameState(GameState):
    players_bitstrings = [random.randint(1, 2**64 - 1) for _ in range(3)]  # 0 is for the empty player
    zobrist_tables = {
        size: [[[random.randint(1, 2**64 - 1) for _ in range(3)] for _ in range(size)] for _ in range(size)]
        for size in range(3, 10)
    }

    def __init__(self, board_size=3, row_length=None, board=None, player=1, board_hash=None):
        self.size = board_size
        self.row_length = row_length if row_length else self.size
        self.board = board
        self.board_hash = board_hash
        self.player = player

        self.zobrist_table = self.zobrist_tables[self.size]
        if self.board is None:
            self.board = [[0] * self.size for _ in range(self.size)]
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

        new_board = [row.copy() for row in self.board]
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
        )
        return new_state

    def skip_turn(self):
        """Used for the null-move heuristic in alpha-beta search"""
        new_board = [row[:] for row in self.board]
        # Pass the same hash since this is only used for null-moves
        return TicTacToeGameState(
            board_size=self.size,
            board=new_board,
            player=3 - self.player,
            row_length=self.row_length,
            board_hash=self.board_hash,
        )

    def get_legal_actions(self):
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == 0]

    def is_terminal(self):
        return self.get_reward(1) != 0 or len(self.get_legal_actions()) == 0

    def get_reward(self, player):
        opponent = 3 - player
        # Check rows
        for i in range(self.size):
            for j in range(self.size - self.row_length + 1):
                if all(self.board[i][j + k] == player for k in range(self.row_length)):
                    return win
                if all(self.board[i][j + k] == opponent for k in range(self.row_length)):
                    return loss

        # Check columns
        for j in range(self.size):
            for i in range(self.size - self.row_length + 1):
                if all(self.board[i + k][j] == player for k in range(self.row_length)):
                    return win
                if all(self.board[i + k][j] == opponent for k in range(self.row_length)):
                    return loss

        # Check diagonal from top-left to bottom-right
        for i in range(self.size - self.row_length + 1):
            for j in range(self.size - self.row_length + 1):
                if all(self.board[i + k][j + k] == player for k in range(self.row_length)):
                    return win
                if all(self.board[i + k][j + k] == opponent for k in range(self.row_length)):
                    return loss

        # Check diagonal from bottom-left to top-right
        for i in range(self.row_length - 1, self.size):
            for j in range(self.size - self.row_length + 1):
                if all(self.board[i - k][j + k] == player for k in range(self.row_length)):
                    return win
                if all(self.board[i - k][j + k] == opponent for k in range(self.row_length)):
                    return loss

        return draw

    def is_capture(self, move):
        # There are no captures in Tic-Tac-Toe, so this function always returns False.
        return False

    def evaluate_move(self, move):
        """
        Evaluates the "freedom" and "connectivity" of a move.

        :param move: The move to evaluate.
        :return: A tuple with the freedom score and connectivity score for the move.
        """
        x, y = move
        adjacent_moves = [(x + i, y + j) for i in [-1, 0, 1] for j in [-1, 0, 1] if i != 0 or j != 0]

        freedom_score = 0
        connectivity_score = 0

        for i, j in adjacent_moves:
            if 0 <= i < self.size and 0 <= j < self.size:
                if self.board[i][j] == 0:
                    freedom_score += 1
                elif self.board[i][j] == self.player:
                    connectivity_score += 1

        return (2 * freedom_score) + connectivity_score

    def visualize(self):
        visual = ""
        for i in range(self.size):
            for j in range(self.size):
                visual += " " + MARKS[self.board[i][j]] + " "
                if j != self.size - 1:
                    visual += "|"
            visual += "\n"
            if i != self.size - 1:
                visual += "-" * (self.size * 4 - 1)
            visual += "\n"
        visual += "hash: " + str(self.board_hash)
        return visual

    @property
    def transposition_table_size(self):
        # return an appropriate size based on the game characteristics
        return 2**16


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


def evaluate_nk_tictactoe(state, player, m_opp_disc: float = 1.0, m_score: float = 1.0):
    def calculate_score(marks, score_factor):
        if opponent not in marks and player in marks:
            return score_factor * ((len(marks) - marks.count(0)) * marks.count(player)) ** 2
        elif player not in marks and opponent in marks:
            return -score_factor * ((len(marks) - marks.count(0)) * marks.count(opponent)) ** 2
        return 0

    def potential_win(marks):
        return (
            marks.count(player if state.player == player else opponent) == state.row_length - 1
            and marks.count(0) == 1
        )

    score = 0
    board = state.board
    size = state.size
    row_length = state.row_length
    opponent = 3 - player

    lines = []

    # Rows and columns
    for i in range(size):
        for j in range(size - row_length + 1):
            lines.append([board[i][j + k] for k in range(row_length)])  # row i
            lines.append([board[j + k][i] for k in range(row_length)])  # column i

    # Diagonals
    for i in range(size - row_length + 1):
        for j in range(size - row_length + 1):
            lines.append([board[i + k][j + k] for k in range(row_length)])  # main diagonal
            lines.append([board[i + k][size - j - k - 1] for k in range(row_length)])  # anti-diagonal

    for marks in lines:
        score += calculate_score(marks, m_score)
        if potential_win(marks):
            return 10000 if state.player == player else -10000

    # Discount score if it is the opponent's turn
    score *= m_opp_disc if state.player == opponent else 1.0

    return score

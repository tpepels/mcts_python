from games.gamestate import GameState


def visualize_tictactoe(state):
    board = state.board
    size = state.size
    visual = ""
    for i in range(size):
        for j in range(size):
            visual += " " + board[i][j] + " "
            if j != size - 1:
                visual += "|"
        visual += "\n"
        if i != size - 1:
            visual += "-" * (size * 4 - 1)
        visual += "\n"
    return visual


class TicTacToeGameState(GameState):
    def __init__(self, size=3, board=None, player=1):
        self.size = size
        self.board = board if board else [[" "] * self.size for _ in range(self.size)]
        self.player = player

    def apply_action(self, action):
        x, y = action
        if self.board[x][y] != " ":
            raise ValueError("Illegal move")

        new_board = [row.copy() for row in self.board]
        new_board[x][y] = "X" if self.player == 1 else "O"
        return TicTacToeGameState(self.size, new_board, 3 - self.player)

    def get_legal_actions(self):
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == " "]

    def is_terminal(self):
        return self.get_reward() != 0 or len(self.get_legal_actions()) == 0

    def get_reward(self):
        for i in range(self.size):
            if all(self.board[i][j] == "X" for j in range(self.size)):
                return 1
            if all(self.board[i][j] == "O" for j in range(self.size)):
                return -1

        for j in range(self.size):
            if all(self.board[i][j] == "X" for i in range(self.size)):
                return 1
            if all(self.board[i][j] == "O" for i in range(self.size)):
                return -1

        if all(self.board[i][i] == "X" for i in range(self.size)):
            return 1
        if all(self.board[i][i] == "O" for i in range(self.size)):
            return -1

        if all(self.board[i][self.size - i - 1] == "X" for i in range(self.size)):
            return 1
        if all(self.board[i][self.size - i - 1] == "O" for i in range(self.size)):
            return -1

        return 0

    def is_capture(self, move):
        # There are no captures in Tic-Tac-Toe, so this function always returns False.
        return False


def evaluate_tictactoe(state, player):
    score = 0
    board = state.board
    size = state.size

    player_mark = "X" if player == 1 else "O"
    opponent_mark = "O" if player == 1 else "X"

    # Check rows and columns
    for i in range(size):
        row_marks = [board[i][j] for j in range(size)]
        col_marks = [board[j][i] for j in range(size)]

        if row_marks.count(player_mark) == size - 1 and " " in row_marks:
            score += 10
        elif row_marks.count(opponent_mark) == size - 1 and " " in row_marks:
            score -= 10
        if col_marks.count(player_mark) == size - 1 and " " in col_marks:
            score += 10
        elif col_marks.count(opponent_mark) == size - 1 and " " in col_marks:
            score -= 10

    # Check diagonals
    main_diag_marks = [board[i][i] for i in range(size)]
    anti_diag_marks = [board[i][size - i - 1] for i in range(size)]

    if main_diag_marks.count(player_mark) == size - 1 and " " in main_diag_marks:
        score += 10
    elif main_diag_marks.count(opponent_mark) == size - 1 and " " in main_diag_marks:
        score -= 10
    if anti_diag_marks.count(player_mark) == size - 1 and " " in anti_diag_marks:
        score += 10
    elif anti_diag_marks.count(opponent_mark) == size - 1 and " " in anti_diag_marks:
        score -= 10

    # Add 1 for every space the player has filled
    for i in range(size):
        for j in range(size):
            if board[i][j] == player_mark:
                score += 1

    return score

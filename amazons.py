from gamestate import GameState


class AmazonsGameState(GameState):
    def __init__(self, board=None, player=1, action=None):
        self.board = board if board is not None else self.initialize_board()
        self.player = player
        self.action = action

    def initialize_board(self):
        board = [[0] * 10 for _ in range(10)]
        for i in (0, 9):
            board[0][i] = board[9][i] = 1
            board[i][0] = board[i][9] = 2
        board[0][3] = board[0][6] = 1
        board[9][3] = board[9][6] = 2
        return board

    def apply_action(self, action):
        x1, y1, x2, y2, x3, y3 = action
        new_board = [row[:] for row in self.board]
        new_board[x2][y2] = new_board[x1][y1]
        new_board[x1][y1] = 0
        new_board[x3][y3] = -1
        return AmazonsGameState(new_board, 3 - self.player, action)

    def get_legal_actions(self):
        actions = []
        for x in range(10):
            for y in range(10):
                if self.board[x][y] == self.player:
                    actions.extend(self.get_legal_moves(x, y))
        return actions

    def get_legal_moves(self, x, y):
        moves = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                while 0 <= nx < 10 and 0 <= ny < 10 and self.board[nx][ny] == 0:
                    moves.append((x, y, nx, ny))
                    nx += dx
                    ny += dy
        return moves

    def is_terminal(self):
        return not self.get_legal_actions()

    def get_reward(self):
        if self.is_terminal():
            return 1 if self.player != 1 else -1
        else:
            return 0


# The evaluate function takes an Amazons game state and the player number (1 or 2) as input.
# It computes the number of legal moves and the number of controlled squares for each player.
# It then calculates the difference between the players for each metric and combines them with equal weights (0.5) to produce an evaluation score.


def evaluate(state, player):
    opponent = 3 - player
    player_moves = 0
    opponent_moves = 0
    player_controlled_squares = 0
    opponent_controlled_squares = 0

    for x in range(10):
        for y in range(10):
            piece = state.board[x][y]

            if piece == player:
                moves = state.get_legal_moves(x, y)
                player_moves += len(moves)
                player_controlled_squares += count_reachable_squares(state, x, y)

            elif piece == opponent:
                moves = state.get_legal_moves(x, y)
                opponent_moves += len(moves)
                opponent_controlled_squares += count_reachable_squares(state, x, y)

    move_difference = (player_moves - opponent_moves) / (player_moves + opponent_moves)
    controlled_squares_difference = (player_controlled_squares - opponent_controlled_squares) / (
        player_controlled_squares + opponent_controlled_squares
    )

    return 0.5 * move_difference + 0.5 * controlled_squares_difference


def count_reachable_squares(state, x, y):
    reachable = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            while 0 <= nx < 10 and 0 <= ny < 10 and state.board[nx][ny] == 0:
                reachable += 1
                nx += dx
                ny += dy
    return reachable


def visualize_amazons(state):
    board = state.board
    for i in range(10):
        row = []
        for j in range(10):
            piece = board[i][j]
            if piece == 1:
                row.append("P")
            elif piece == 2:
                row.append("Q")
            elif piece == -1:
                row.append("A")
            else:
                row.append(".")
        print(" ".join(row))
    print("\n")

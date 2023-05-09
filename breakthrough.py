from gamestate import GameState


def visualize_breakthrough(state):
    board = state.board
    for i in range(8):
        row = []
        for j in range(8):
            piece = board[i][j]
            if piece == 1:
                row.append("P")
            elif piece == 2:
                row.append("Q")
            else:
                row.append(".")
        print(" ".join(row))
    print("\n")


class BreakthroughGameState(GameState):
    def __init__(self, board=None, player=1):
        if board is None:
            self.board = [0] * 64
            for i in range(16):
                self.board[i] = 200 + i
                self.board[48 + i] = 100 + i
        else:
            self.board = board
        self.player = player

    def apply_action(self, action):
        from_position, to_position = action
        piece = self.board[from_position]
        self.board[from_position] = 0
        self.board[to_position] = piece
        return BreakthroughGameState(self.board, 3 - self.player)

    def get_legal_actions(self):
        legal_actions = []

        for position, piece in enumerate(self.board):
            if piece == 0 or piece // 100 != self.player:
                continue

            row, col = divmod(position, 8)
            dr = -1 if self.player == 1 else 1

            for dc in (-1, 0, 1):
                new_row, new_col = row + dr, col + dc
                if not in_bounds(new_row, new_col):
                    continue

                new_position = new_row * 8 + new_col
                if self.board[new_position] == 0 or self.board[new_position] // 100 != self.player:
                    legal_actions.append((position, new_position))

        return legal_actions

    def is_terminal(self):
        return any(
            (piece // 100 == self.player and (piece % 100) // 8 == (7 if self.player == 1 else 0))
            for piece in self.board
        )

    def get_reward(self):
        if self.is_terminal():
            return 1 if self.player == 1 else -1
        return 0

    def in_bounds(self, r, c):
        return 0 <= r < 8 and 0 <= c < 8


# The evaluate function takes a Breakthrough game state and the player number (1 or 2) as input.
# It computes the number of pieces for each player, the average distance of the pieces to the opponent's side,
# and the number of pieces that are close to the opponent's side.
# It then calculates the difference between the players for each metric and combines them
# with weights to produce an evaluation score.


def evaluate(state, player):
    opponent = 3 - player
    player_pieces = 0
    opponent_pieces = 0
    player_distance = 0
    opponent_distance = 0
    player_near_opponent_side = 0
    opponent_near_opponent_side = 0

    for x in range(8):
        for y in range(8):
            piece = state.board[x][y]

            if piece == player:
                player_pieces += 1
                player_distance += 7 - x if player == 1 else x

                if player == 1 and x >= 6:
                    player_near_opponent_side += 1
                elif player == 2 and x <= 1:
                    player_near_opponent_side += 1

            elif piece == opponent:
                opponent_pieces += 1
                opponent_distance += 7 - x if opponent == 1 else x

                if opponent == 1 and x >= 6:
                    opponent_near_opponent_side += 1
                elif opponent == 2 and x <= 1:
                    opponent_near_opponent_side += 1

    if player_pieces == 0 or opponent_pieces == 0:
        return 0

    piece_difference = (player_pieces - opponent_pieces) / (player_pieces + opponent_pieces)
    avg_distance_difference = (opponent_distance - player_distance) / (player_pieces + opponent_pieces)
    near_opponent_side_difference = (player_near_opponent_side - opponent_near_opponent_side) / (
        player_pieces + opponent_pieces
    )

    return 0.4 * piece_difference + 0.4 * avg_distance_difference + 0.2 * near_opponent_side_difference


# Lorenz evaluation functions


def in_bounds(x, y):
    return 0 <= x < 8 and 0 <= y < 8


def is_safe(x, y, player):
    row_offsets = [-1, -1, 1, 1]
    col_offsets = [-1, 1, -1, 1]
    attackers = 0
    defenders = 0

    for i in range(4):
        new_x = x + row_offsets[i]
        new_y = y + col_offsets[i]

        if in_bounds(new_x, new_y):
            piece = state.board[new_x][new_y]

            if player == 1:
                if i < 2 and piece == opponent:
                    attackers += 1
                elif i >= 2 and piece == player:
                    defenders += 1
            else:
                if i < 2 and piece == player:
                    defenders += 1
                elif i >= 2 and piece == opponent:
                    attackers += 1

    return attackers <= defenders


lorentz_values = [
    5,
    15,
    15,
    5,
    5,
    15,
    15,
    5,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    2,
    4,
    6,
    6,
    6,
    6,
    6,
    6,
    4,
    7,
    10,
    10,
    10,
    10,
    10,
    10,
    7,
    11,
    15,
    15,
    15,
    15,
    15,
    15,
    11,
    16,
    21,
    21,
    21,
    21,
    21,
    21,
    16,
    20,
    28,
    28,
    28,
    28,
    28,
    28,
    20,
    36,
    36,
    36,
    36,
    36,
    36,
    36,
    36,
]


def lorenz_evaluation(state, player):
    opponent = 3 - player

    p1eval = 0
    p2eval = 0

    for x in range(8):
        for y in range(8):
            piece = state.board[x][y]

            if piece == player:
                base_value = lorentz_values[x * 8 + y]
                p1eval += base_value

                if is_safe(x, y, player):
                    p1eval += 0.5 * base_value

            elif piece == opponent:
                base_value = lorentz_values[(7 - x) * 8 + y]
                p2eval += base_value

                if is_safe(x, y, opponent):
                    p2eval += 0.5 * base_value

    if player == 1:
        eval_value = p1eval - p2eval
    else:
        eval_value = p2eval - p1eval

    return eval_value


def lorenz_enhanced_evaluation(state, player):
    def piece_mobility(position, player):
        row, col = divmod(position, 8)
        mobility = 0

        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            new_row, new_col = row + dr, col + dc
            if in_bounds(new_row, new_col):
                new_position = new_row * 8 + new_col
                if state[new_position] == 0 or state[new_position] // 100 != player:
                    mobility += 1

        return mobility

    def is_endgame(state):
        pieces_count = sum(1 for cell in state if cell != 0)
        return pieces_count <= 16  # You can adjust this value based on your preference

    board_values = 0
    mobility_values = 0
    for position, piece in enumerate(state):
        if piece == 0:
            continue

        piece_player = piece // 100
        piece_value = lorentz_values[position] if piece_player == 2 else lorentz_values[63 - position]

        if piece_player == player:
            board_values += piece_value
            mobility_values += piece_mobility(position, player)
        else:
            board_values -= piece_value

        if is_safe(position, position, piece_player):
            safety_bonus = 0.5 * piece_value
            board_values += safety_bonus if piece_player == player else -safety_bonus

    if is_endgame(state):
        endgame_bonus = sum(1 for piece in state if piece != 0 and piece // 100 == player) * 10
        board_values += endgame_bonus

    return board_values + mobility_values

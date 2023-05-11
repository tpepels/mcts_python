from games.gamestate import GameState


def visualize_kalah(state):
    top_row = " ".join(
        [str(state.board[i]).rjust(2) for i in range(state.num_houses + 1, len(state.board))][::-1]
    )

    bottom_row = " ".join([str(state.board[i]).rjust(2) for i in range(state.num_houses)])
    player1_store = str(state.board[state.num_houses]).rjust(2)
    player2_store = str(state.board[-1]).rjust(2)

    return f"Player 2: {player2_store}\n{top_row}\n{bottom_row}\nPlayer 1: {player1_store}"


class KalahGameState(GameState):
    def __init__(self, board=None, player=1, num_houses=6, num_seeds=4):
        """
        Initialize the Kalah game state.

        :param num_houses: The number of houses per player (default: 6).
        :param num_seeds: The initial number of seeds per house (default: 4).
        :param board: Optional board state, represented as a list of integers (default: None).
        :param player: The player whose turn it is, 1 for player 1 and 2 for player 2 (default: 1).
        """
        if board is None:
            board = [num_seeds] * (num_houses * 2 + 2)
            board[num_houses] = 0
            board[-1] = 0
        self.board = board
        self.player = player
        self.num_houses = num_houses

    def apply_action(self, action):
        """
        Apply an action to the current game state and return the resulting new state.
        The state of the instance is not altered in this method.

        :param action: The action to apply, represented as the index of the house to pick up seeds from.
        :return: The resulting new game state after applying the action.
        """
        if not self._is_valid_move(action):
            raise ValueError("Invalid move")

        new_board, next_player = self._sow_seeds(action)

        # Capture opponent's seeds if the last seed is sown in an empty house on the player's side and the opposite house has seeds.
        last_index = (action + new_board[action]) % len(new_board)
        if (self.player == 1 and 0 <= last_index <= 5) or (self.player == 2 and 7 <= last_index <= 12):
            if new_board[last_index] == 1:
                opposite_house = self._opposite_house(last_index)
                if new_board[opposite_house] > 0:
                    # Capture seeds
                    captured_seeds = new_board[opposite_house] + 1
                    new_board[opposite_house] = 0
                    new_board[last_index] = 0
                    # Add captured seeds to the player's store
                    if self.player == 1:
                        new_board[6] += captured_seeds
                    else:
                        new_board[13] += captured_seeds

        # Check if one side is empty and move all seeds on the other side to the respective store
        p1_houses_empty = all(new_board[i] == 0 for i in range(0, 6))
        p2_houses_empty = all(new_board[i] == 0 for i in range(7, 13))

        if p1_houses_empty or p2_houses_empty:
            if p1_houses_empty:
                new_board[13] += sum(new_board[i] for i in range(7, 13))
                for i in range(7, 13):
                    new_board[i] = 0
            else:
                new_board[6] += sum(new_board[i] for i in range(0, 6))
                for i in range(0, 6):
                    new_board[i] = 0

        new_state = KalahGameState(new_board, next_player)
        return new_state

    def get_legal_actions(self):
        """
        Get a list of legal actions for the current game state.

        :return: A list of legal actions, represented as the indices of the houses with seeds.
        """
        start = 0 if self.player == 1 else self.num_houses + 1
        end = self.num_houses if self.player == 1 else len(self.board) - 1
        return [i for i in range(start, end) if self.board[i] > 0]

    def is_terminal(self):
        """
        Check if the current game state is a terminal state, i.e., the game has ended.

        :return: True if the state is terminal, False otherwise.
        """
        player1_houses_empty = all(self.board[i] == 0 for i in range(0, self.num_houses))
        player2_houses_empty = all(
            self.board[i] == 0 for i in range(self.num_houses + 1, len(self.board) - 1)
        )
        return player1_houses_empty or player2_houses_empty

    def get_reward(self):
        """
        The reward is 1 for player 1 if they have won, -1 for player 2 if they have won, and 0 otherwise.

        :return: The reward value.
        """
        if not self.is_terminal():
            return 0

        player1_score = self.board[self.num_houses]
        player2_score = self.board[-1]

        if player1_score > player2_score:
            return 1
        elif player1_score < player2_score:
            return -1
        else:
            return 0

    def _sow_seeds(self, index):
        """
        Sow the seeds from the specified house in a counter-clockwise direction.

        :param index: The index of the house to pick up seeds from.
        :return: The updated board after sowing seeds and the player whose turn it is next.
        """
        board_copy = self.board.copy()
        seeds = board_copy[index]
        board_copy[index] = 0

        next_player = self.player

        while seeds > 0:
            index = (index + 1) % len(board_copy)

            # Skip the opponent's store
            if (self.player == 1 and index == 13) or (self.player == 2 and index == 6):
                continue

            board_copy[index] += 1
            seeds -= 1

            # If the last seed is sown in the player's store, it's their turn again
            if seeds == 0 and ((self.player == 1 and index == 6) or (self.player == 2 and index == 13)):
                next_player = self.player
            else:
                next_player = 3 - self.player

        return board_copy, next_player

    def _opposite_house(self, house):
        """
        Get the index of the house opposite to the given house index.

        :param house: The index of the house.
        :return: The index of the opposite house.
        """
        if 0 <= house <= 5:
            return 12 - house
        elif 7 <= house <= 12:
            return 12 - (house - 7)
        else:
            raise ValueError("Invalid house index")

    def _is_valid_move(self, index):
        """
        Check if the given move is valid for the current player.

        :param index: The index of the house to pick up seeds from.
        :return: True if the move is valid, False otherwise.
        """
        if self.player == 1 and 0 <= index <= 5 and self.board[index] > 0:
            return True
        elif self.player == 2 and 7 <= index <= 12 and self.board[index] > 0:
            return True
        return False


def simple_evaluation_function(state: KalahGameState, player: int) -> float:
    if player == 1:
        return state.board[0] - state.board[7]
    else:
        return state.board[7] - state.board[0]


def sophisticated_evaluation_function(state: KalahGameState, player: int) -> float:
    if player == 1:
        opponent = 2
        player_houses = range(1, 7)
        opponent_houses = range(8, 14)
    else:
        opponent = 1
        player_houses = range(8, 14)
        opponent_houses = range(1, 7)

    # Difference in seeds in stores
    store_diff = state.board[player * 7 - 1] - state.board[opponent * 7 - 1]

    # Difference in seeds on each player's side
    player_seeds = sum(state.board[i] for i in player_houses)
    opponent_seeds = sum(state.board[i] for i in opponent_houses)
    seeds_diff = player_seeds - opponent_seeds

    # The number of empty houses on the opponent's side
    empty_opponent_houses = sum(1 for i in opponent_houses if state.board[i] == 0)

    # Weights for each factor
    w1, w2, w3 = 1.0, 0.5, 0.25

    evaluation = w1 * store_diff + w2 * seeds_diff + w3 * empty_opponent_houses

    return evaluation


def evaluate_kalah_state(state: KalahGameState, player: int, weights: dict) -> float:
    # * Source: https://digitalcommons.andrews.edu/cgi/viewcontent.cgi?article=1259&context=honors
    opponent = 3 - player

    # Heuristic 1: Chain moves
    def chain_moves(state, player):
        chain_count = 0
        for pit in state.get_legal_actions():
            new_state = state.apply_action(pit)
            if new_state.current_player == player:
                chain_count += 1
        return chain_count

    # Heuristic 2: Capture seeds
    def capture_seeds(state, player):
        return state.board[state.get_store_index(player)] - state.previous_capture_count

    # Heuristic 3: Limit opponent score
    def limit_opponent_score(state, player):
        opponent_score = state.board[state.get_store_index(opponent)]
        return -opponent_score

    # Heuristic 4: Store difference
    def store_difference(state, player):
        return state.board[state.get_store_index(player)] - state.board[state.get_store_index(opponent)]

    # Heuristic 5: Available moves
    def available_moves(state, player):
        return len(state.get_legal_actions())

    score = (
        weights["chain_moves"] * chain_moves(state, player)
        + weights["capture_seeds"] * capture_seeds(state, player)
        + weights["limit_opponent_score"] * limit_opponent_score(state, player)
        + weights["store_difference"] * store_difference(state, player)
        + weights["available_moves"] * available_moves(state, player)
    )

    return score

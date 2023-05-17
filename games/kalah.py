from games.gamestate import GameState


def visualize_kalah(state):
    """
    Visualize the current game state.
    """
    top_row = " ".join(
        [str(state.board[i]).rjust(2) + f" ({i})" for i in range(state.num_houses - 1, -1, -1)]
    )

    bottom_row = " ".join(
        [
            str(state.board[i]).rjust(2) + f" ({i})"
            for i in range(state.num_houses + 1, 2 * state.num_houses + 1)
        ]
    )

    player2_store = str(state.board[-1]).rjust(2)
    player1_store = str(state.board[state.num_houses]).rjust(2)

    return (
        f"Player 1's store: {player1_store}\n"
        f"{top_row}\n"
        f"{bottom_row}\n"
        f"Player 2's store: {player2_store}"
    )


class KalahGameState(GameState):
    def __init__(self, board=None, player=1):
        """
        Initialize the Kalah game state.

        :param num_houses: The number of houses per player (default: 6).
        :param num_seeds: The initial number of seeds per house (default: 4).
        :param board: Optional board state, represented as a list of integers (default: None).
        :param player: The player whose turn it is, 1 for player 1 and 2 for player 2 (default: 1).
        """
        if board is None:
            board = [4] * 14  # 12 houses plus two pits
            board[6] = 0  # player 1's pit
            board[-1] = 0  # player 2's pit
        self.board = board
        self.player = player
        self.num_houses = 6

    def apply_action(self, action):
        """
        Apply an action to the current game state and return the resulting new state.
        The state of the instance is not altered in this method.

        :param action: The action to apply, represented as the index of the house to pick up seeds from.
        :return: The resulting new game state after applying the action.
        """
        if not self._is_valid_move(action):
            raise ValueError("Invalid move")

        new_board, next_player, last_index = self._sow_seeds(action)

        cap = False

        # Capture opponent's seeds if the last seed is sown in an empty house on the player's side and the opposite house has seeds.
        if (self.player == 1 and 0 <= last_index <= 5) or (self.player == 2 and 7 <= last_index <= 12):
            if new_board[last_index] == 1:
                opposite_house = 12 - last_index
                if new_board[opposite_house] > 0:
                    if not self.is_capture(action):
                        print("Capture but no is_capture")
                        print(visualize_kalah(self))
                        print(action)
                        self.is_capture(action, True)
                        return None

                    # Capture seeds
                    cap = True
                    captured_seeds = new_board[opposite_house] + 1
                    new_board[opposite_house] = 0
                    new_board[last_index] = 0
                    # Add captured seeds to the player's store
                    if self.player == 1:
                        new_board[6] += captured_seeds
                    else:
                        new_board[13] += captured_seeds

        if cap == False and self.is_capture(action):
            print("No capture but is_capture!")
            print(visualize_kalah(self))
            print(action)
            self.is_capture(action, True)
            return None

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
        # if new_state.is_terminal():
        #     print(f"Move {action} by {self.player} leads to a terminal position!")
        #     print(visualize_kalah(self))
        #     print("Resulting in gamestate:")
        #     print(visualize_kalah(new_state))
        #     print("-==-" * 6)

        # if self.player == next_player:
        #     print(f"Move {action} by {self.player} leads another move yay!")
        #     print(visualize_kalah(self))
        #     print("Resulting in gamestate:")
        #     print(visualize_kalah(new_state))
        #     print("-==-" * 6)

        # if cap:
        #     print(f"Move {action} by {self.player} leads to capture!")
        #     print(visualize_kalah(self))
        #     print("Resulting in gamestate:")
        #     print(visualize_kalah(new_state))
        #     print("-==-" * 6)
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

        return board_copy, next_player, index

    def _is_valid_move(self, index):
        """
        Check if the given move is valid for the current player.

        :param index: The index of the house to pick up seeds from.
        :return: True if the move is valid, False otherwise.
        """

        if self.is_terminal():  # Stop moving in a terminal state
            return False

        return (self.player == 1 and 0 <= index <= 5 and self.board[index] > 0) or (
            self.player == 2 and 7 <= index <= 12 and self.board[index] > 0
        )

    def is_capture(self, move, _print=False):
        """
        Check if the given move results in a capture.

        :param move: The move to check, represented as the index of the house to pick up seeds from.
        :return: True if the move results in a capture, False otherwise.
        """
        if not self._is_valid_move(move):
            raise ValueError("Invalid move")

        seeds = self.board[move]

        # Calculate how often the opponent's pit is passed
        size = len(self.board)
        passes = count_passes(
            seeds=seeds,
            move=move,
            k=6 if self.player == 2 else 13,
            size=size,
        )
        # Calculate total steps considering the opponent's store
        total_steps = move + seeds + passes
        last_index = total_steps % (13 + passes)  # And, finally the index where the last stone is dropped

        opp = 12 - last_index
        passed_opp = total_steps > opp if move < opp else total_steps > size + opp
        rounds = seeds // 13
        # if we end up where we started, that means that we are in a house with 0 seeds, but the board still
        # has the seeds in it.
        if ((self.board[last_index] == 0 and rounds == 0) or last_index == move) and (
            (self.player == 1 and 0 <= last_index <= 5) or (self.player == 2 and 7 <= last_index <= 12)
        ):
            # Either the opposing side had a stone before, or we dropped one in it
            if self.board[opp] > 0 or passed_opp:
                if _print:
                    print(
                        f"YES capture {move=} {seeds=} {last_index=} {total_steps=} {passes=} {passed_opp=}"
                    )
                return True
        if _print:
            print(f"NO capture {move=} {seeds=} {last_index=} {total_steps=} {passes=} {passed_opp=}")
        return False


def count_passes(seeds, move, k, size=13):
    """
    Function can be used to determine the last index of a given move without having to execute the full move.

    Args:
        seeds (int): the number of seeds picked up
        size (int): the size of the board (default = 13)
        move (int): The position the seeds are picked up from
        k (int): The function counts how often position k is passed (6 for p1 13 for p2 if we count the stores)

    Returns:
        int: How often k is passed given the parameters
    """
    rounds = seeds // size
    remaining_steps = seeds % size

    # if (move >= k and (remaining_steps + move) % size >= k) or (
    #     move < k and remaining_steps + move >= size + k
    # ):
    if remaining_steps + move >= size + k:
        rounds += 1
    return rounds


def simple_evaluation_function(state: KalahGameState, player: int) -> float:
    if player == 1:
        return state.board[0] - state.board[7]
    else:
        return state.board[7] - state.board[0]


def enhanced_evaluation_function(state: KalahGameState, player: int) -> float:
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

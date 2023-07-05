from games.gamestate import GameState, normalize, win, loss, draw
import random


MAX_SEEDS = 72  # maximum number of seeds in one position, it's 72 as it's the total seeds in the game


class KalahGameState(GameState):
    players_bitstrings = [random.randint(1, 2**64 - 1) for _ in range(3)]  # 0 is for the empty player
    zobrist_table = [
        [random.randint(1, 2**64 - 1) for _ in range(MAX_SEEDS)] for _ in range(14)
    ]  # 14 slots

    def __init__(self, board=None, player=1):
        """
        Initialize the Kalah game state.

        :param num_houses: The number of houses per player (default: 6).
        :param num_seeds: The initial number of seeds per house (default: 4).
        :param board: Optional board state, represented as a list of integers (default: None).
        :param player: The player whose turn it is, 1 for player 1 and 2 for player 2 (default: 1).
        """
        self.player = player
        self.num_houses = 6
        if board is None:
            self.board = [4] * 14  # 12 houses plus two pits
            self.board[6] = 0  # player 1's pit
            self.board[-1] = 0  # player 2's pit
        else:
            self.board = board

        self.board_hash = 0
        for position in range(14):
            seeds = self.board[position]
            self.board_hash ^= self.zobrist_table[position][seeds]
        self.board_hash ^= self.players_bitstrings[
            self.player
        ]  # XOR with the bitstring of the current player

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

        # Capture opponent's seeds if the last seed is sown in an empty house on the player's side and the opposite house has seeds.
        if (self.player == 1 and 0 <= last_index <= 5) or (self.player == 2 and 7 <= last_index <= 12):
            if new_board[last_index] == 1:
                opposite_house = 12 - last_index
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
                for i in range(7, 13):
                    new_board[13] += new_board[i]
                    new_board[i] = 0
            else:
                for i in range(0, 6):
                    new_board[6] += new_board[i]
                    new_board[i] = 0

        new_state = KalahGameState(new_board, next_player)
        return new_state

        # Uncomment this for debugging the game
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
        # return new_state

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

    def skip_turn(self):
        """Used for the null-move heuristic in alpha-beta search

        Returns:
            BreakthroughGameState: A new gamestate in which the players are switched but no move performed
        """
        new_board = self.board.copy()
        return KalahGameState(new_board, 3 - self.player)

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

    def get_reward(self, player):
        """
        Returns the reward value of a terminal state [-1, 0, 1] (loss/draw/win)

        :return: The reward value.
        """
        if not self.is_terminal():
            return 0

        player1_score = self.board[self.num_houses]
        player2_score = self.board[-1]

        if player1_score > player2_score:
            return win if player == 1 else loss
        elif player1_score < player2_score:
            return loss if player == 1 else win
        else:
            return draw

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

    def is_capture(self, move):
        """
        Check if the given move results in a capture.

        :param move: The move to check, represented as the index of the house to pick up seeds from.
        :return: True if the move results in a capture, False otherwise.
        """
        seeds = self.board[move]

        # Calculate how often the opponent's pit is passed
        size = len(self.board)
        last_index, total_steps = calc_last_index_total_steps(
            seeds=seeds,
            move=move,
            k=6 if self.player == 2 else 13,
            size=size,
        )
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
                return True
        return False

    def evaluate_move(self, move):
        """
        Evaluates the given move using a heuristic based on the potential benefits of the move.

        :param move: The move to evaluate.
        :return: The heuristic score for the move.
        """

        # Initialize score to 0
        score = 0

        # Check if the move results in a capture and assign a positive score
        if self.is_capture(move):
            score += 2

        # Check if the move results in another move for the current player and assign a positive score
        last_index, _ = calc_last_index_total_steps(self.board[move], move, 13 if self.player == 1 else 6)
        if (self.player == 1 and last_index == 6) or (self.player == 2 and last_index == 6):
            score += 1

        return score

    def visualize(self):
        """
        Visualize the current game state.
        """
        top_row = " ".join(
            [str(self.board[i]).rjust(2) + f" ({i})" for i in range(self.num_houses - 1, -1, -1)]
        )

        bottom_row = " ".join(
            [
                str(self.board[i]).rjust(2) + f" ({i})"
                for i in range(self.num_houses + 1, 2 * self.num_houses + 1)
            ]
        )

        player2_store = str(self.board[-1]).rjust(2)
        player1_store = str(self.board[self.num_houses]).rjust(2)

        return (
            f"Player 1's store: {player1_store}\n"
            f"{top_row}\n"
            f"{bottom_row}\n"
            f"Player 2's store: {player2_store}\n"
            f"hash: {self.board_hash}"
        )

    @property
    def transposition_table_size(self):
        # return an appropriate size based on the game characteristics
        return 2**21


def calc_last_index_total_steps(seeds, move, k, size=13):
    passes = count_passes(
        seeds=seeds,
        move=move,
        k=k,
        size=size,
    )
    # Calculate total steps considering the opponent's store
    total_steps = move + seeds + passes
    return total_steps % (13 + passes), total_steps


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


def evaluate_kalah_simple(
    state: KalahGameState,
    player: int,
    m_opp_disc: float = 0.9,
) -> float:
    if player == 1:
        return state.board[6] - state.board[-1] * (m_opp_disc if state.player == 3 - player else 1)
    else:
        return state.board[-1] - state.board[6] * (m_opp_disc if state.player == 3 - player else 1)


def evaluate_kalah_enhanced(
    state: KalahGameState,
    player: int,
    m_score: float = 1.0,
    m_seed_diff: float = 0.5,
    m_empty: float = 0.25,
    m_double: float = 0.5,
    m_capture: float = 0.5,
    m_opp_disc: float = 0.9,
    a: int = 5,
    norm: bool = True,
) -> float:
    """
    Evaluate a given state of a Kalah game from the perspective of the specified player.

    :param state: The game state to evaluate.
    :param player: The player for whom the evaluation is being done.
    :param m_simple_eval: The weight for the simple evaluation function.
    :param m_seed_diff: The weight for the difference in seeds between the players.
    :param m_empty_houses_diff: The weight for the difference in empty houses between the players.
    :param m_double_moves_diff: The weight for the difference in potential double moves between the players.
    :param m_capture_moves_diff: The weight for the difference in potential capture moves between the players.
    :param a: A parameter used in normalization.
    :param norm: A boolean indicating whether to normalize the evaluation.
    :return: A score representing the player's advantage in the game state.
    """
    if player == 1:
        player_store = 6
        opponent_store = 13
        player_houses = range(0, 6)
    else:
        player_store = 13
        opponent_store = 6
        player_houses = range(7, 13)

    player_seeds = opponent_seeds = 0
    player_double_moves = opponent_double_moves = 0
    player_capture_moves = opponent_capture_moves = 0
    empty_opponent_houses = empty_player_houses = 0

    for i in range(0, 13):  # go through all houses (excluding p2 store)
        if i == 6:  # Skip p1 store
            continue

        seeds = state.board[i]
        # Check if the house belongs to the player or the opponent
        if i in player_houses:
            player_seeds += seeds
            if seeds == 0:
                empty_player_houses += 1
            else:
                if m_double != 0 and calc_last_index_total_steps(seeds, i, opponent_store)[0] == player_store:
                    player_double_moves += 1
                elif m_capture != 0 and is_capture(state.board, i, player):
                    player_capture_moves += 1

        else:  # This is possible because we exclude the stores from the loop
            opponent_seeds += seeds
            if seeds == 0:
                empty_opponent_houses += 1
            else:
                if m_double != 0 and calc_last_index_total_steps(seeds, i, player_store)[0] == opponent_store:
                    opponent_double_moves += 1
                elif m_capture != 0 and is_capture(state.board, i, 3 - player):
                    opponent_capture_moves += 1

    evaluation = (
        m_score * evaluate_kalah_simple(state, player, m_opp_disc=1.0)  # Don't apply the discount twice
        + m_seed_diff * (player_seeds - opponent_seeds)
        + m_empty * (empty_opponent_houses - empty_player_houses)
        + m_double * (player_double_moves - opponent_double_moves)
        + m_capture * (player_capture_moves - opponent_capture_moves)
    ) * (m_opp_disc if state.player == 3 - player else 1)

    if norm:
        return normalize(evaluation, a)
    else:
        return evaluation


def is_capture(board, move, player):
    """
    Check if the given move results in a capture.

    :param move: The move to check, represented as the index of the house to pick up seeds from.
    :return: True if the move results in a capture, False otherwise.
    """
    seeds = board[move]

    # Calculate how often the opponent's pit is passed
    size = len(board)
    last_index, total_steps = calc_last_index_total_steps(
        seeds=seeds,
        move=move,
        k=6 if player == 2 else 13,
        size=size,
    )
    opp = 12 - last_index
    passed_opp = total_steps > opp if move < opp else total_steps > size + opp
    rounds = seeds // 13

    # If we end up where we started, that means that we are in a house with 0 seeds, but the board still
    # has the seeds in it.
    if ((board[last_index] == 0 and rounds == 0) or last_index == move) and (
        (player == 1 and 0 <= last_index <= 5) or (player == 2 and 7 <= last_index <= 12)
    ):
        # Either the opposing side had a stone before, or we dropped one in it
        if board[opp] > 0 or passed_opp:
            return True
    return False

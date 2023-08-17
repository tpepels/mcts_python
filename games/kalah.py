# cython: language_level=3
import cython
from games.gamestate import GameState, win, loss, draw
from c_util import normalize
import random

if cython.compiled:
    print("Kalah is compiled.")
else:
    print("Kalah is just a lowly interpreted script.")

MAX_SEEDS = 72  # maximum number of seeds in one position, it's 72 as it's the total seeds in the game


class KalahGameState(GameState):
    players_bitstrings = [random.randint(1, 2**60 - 1) for _ in range(3)]  # 0 is for the empty player
    zobrist_table = [
        [random.randint(1, 2**60 - 1) for _ in range(MAX_SEEDS)] for _ in range(14)
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
        action = action[0]
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

    def get_random_action(self):
        return random.choice(self.get_legal_actions())

    def yield_legal_actions(self):
        """
        Generate legal actions for the current game state.

        :yield: Legal actions, represented as the indices of the houses with seeds.
        """
        start = 0 if self.player == 1 else self.num_houses + 1
        end = self.num_houses if self.player == 1 else len(self.board) - 1
        actions = [i for i in range(start, end) if self.board[i] > 0]

        for action in actions:
            yield (action,)

    def get_legal_actions(self):
        """
        Get a list of legal actions for the current game state.

        :return: A list of legal actions, represented as the indices of the houses with seeds.
        """
        start = 0 if self.player == 1 else self.num_houses + 1
        end = self.num_houses if self.player == 1 else len(self.board) - 1
        return [(i,) for i in range(start, end) if self.board[i] > 0]

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

    def is_capture(self, move):
        """
        Check if the given move results in a capture.

        :param move: The move to check, represented as the index of the house to pick up seeds from.
        :return: True if the move results in a capture, False otherwise.
        """
        return is_capture(self.board, move, self.player)
        # seeds = self.board[move]

        # # Calculate how often the opponent's pit is passed
        # size = len(self.board)
        # last_index, total_steps = calc_last_index_total_steps(
        #     seeds=seeds,
        #     move=move,
        #     k=6 if self.player == 2 else 13,
        #     size=size,
        # )
        # opp = 12 - last_index
        # passed_opp = total_steps > opp if move < opp else total_steps > size + opp
        # rounds = seeds // 13
        # # if we end up where we started, that means that we are in a house with 0 seeds, but the board still
        # # has the seeds in it.
        # if ((self.board[last_index] == 0 and rounds == 0) or last_index == move) and (
        #     (self.player == 1 and 0 <= last_index <= 5) or (self.player == 2 and 7 <= last_index <= 12)
        # ):
        #     # Either the opposing side had a stone before, or we dropped one in it
        #     if self.board[opp] > 0 or passed_opp:
        #         return True

        # return False

    def evaluate_moves(self, moves):
        """
        Evaluate the given moves using a heuristic based on the potential benefits of the move.

        :param moves: The list of moves to evaluate.
        :return: The list of heuristic scores for the moves.
        """
        scores: list[tuple] = [()] * len(moves)
        for i in range(len(moves)):
            scores[i] = (moves[i], evaluate_move(self.board, moves[i][0], self.player))
        return scores

    def move_weights(self, moves):
        """
        Evaluate the given moves using a heuristic based on the potential benefits of the move.

        :param moves: The list of moves to evaluate.
        :return: The list of heuristic scores for the moves.
        """
        scores: list[int] = [0] * len(moves)
        for i in range(len(moves)):
            scores[i] = evaluate_move(self.board, moves[i][0], self.player)
        return scores

    def evaluate_move(self, move):
        """
        Evaluates the given move using a heuristic based on the potential benefits of the move.

        :param move: The move to evaluate.
        :return: The heuristic score for the move.
        """
        return evaluate_move(self.board, move[0], self.player)

    def visualize(self, full_debug=False):
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
        output = (
            f"Player 1's store: {player1_store}\n"
            f"{top_row}\n"
            f"{bottom_row}\n"
            f"Player 2's store: {player2_store}\n"
        )
        output += "..." * 60 + "\n"

        if full_debug:
            actions = self.get_legal_actions()
            output += f"Player: {self.player} | {len(actions)} actions: {[a[0] for a in actions]} | hash: {self.board_hash}\n"
            output += f"Reward: {self.get_reward(1)}/{self.get_reward(2)} | Terminal?: {self.is_terminal()}\n"
            # ------------------ Evaluation ----------------------
            output += "-*-" * 8 + "\n"
            simple_eval_1 = evaluate_kalah_simple(self, 1)
            simple_eval_2 = evaluate_kalah_simple(self, 2)
            output += f"Simple Eval P1: {simple_eval_1} P2: {simple_eval_2}\n"
            # --------------- Simple Evaluation -------------------
            enhanced_eval_1 = evaluate_kalah_enhanced(self, 1)
            enhanced_eval_2 = evaluate_kalah_enhanced(self, 2)
            output += f"Enhanced Eval P1: {enhanced_eval_1} P2: {enhanced_eval_2}\n"
            output += "-*-" * 8 + "\n"

            for action in actions:
                output += f"{action} is capture? {is_capture(self.board, action[0], self.player)}\n"

            if len(actions) > 0:
                actions = self.evaluate_moves(self.get_legal_actions())
                actions = sorted(actions, key=lambda x: x[1], reverse=True)
                output += "..." * 60 + "\n"
                output += str(actions)

        return output

    @property
    def transposition_table_size(self):
        # return an appropriate size based on the game characteristics
        return 2**17

    def __repr__(self) -> str:
        return "kalah"


@cython.ccall
@cython.locals(move=cython.int, score=cython.int, last_index=cython.int, board=cython.list, player=cython.int)
def evaluate_move(board, move, player):
    """
    Evaluates the given move using a heuristic based on the potential benefits of the move.

    :param move: The move to evaluate.
    :return: The heuristic score for the move.
    """

    # Initialize score to the number of seeds in the house
    score = board[move]
    # Check if the move results in a capture and assign a positive score
    if is_capture(board, move, player):
        last_index, _ = calc_last_index_total_steps(
            seeds=board[move],
            move=move,
            k=6 if player == 2 else 13,
            size=len(board),
        )
        score += board[12 - last_index] * 4
    # Check if the move results in another move for the current player and assign a positive score
    last_index, _ = calc_last_index_total_steps(board[move], move, 13 if player == 1 else 6)
    if (player == 1 and last_index == 6) or (player == 2 and last_index == 6):
        score += 10

    return score


@cython.ccall
@cython.locals(
    seeds=cython.int,
    move=cython.int,
    k=cython.int,
    size=cython.int,
    passes=cython.int,
    remaining_steps=cython.int,
    total_steps=cython.int,
)
def calc_last_index_total_steps(seeds, move, k, size=13):
    passes = seeds // size
    remaining_steps = seeds % size

    if remaining_steps + move >= size + k:
        passes += 1

    # Calculate total steps considering the opponent's store
    total_steps = move + seeds + passes
    return total_steps % (13 + passes), total_steps


@cython.ccall
@cython.locals(
    state=cython.object,
    player=cython.int,
    m_opp_disc=cython.double,
    a=cython.int,
    norm=cython.bint,
    score=cython.double,
)
def evaluate_kalah_simple(state, player, m_opp_disc=0.9, a=20, norm=0):
    score = 0
    if player == 1:
        score = (state.board[6] - state.board[-1]) * (m_opp_disc if state.player == 3 - player else 1)
    else:
        score = (state.board[-1] - state.board[6]) * (m_opp_disc if state.player == 3 - player else 1)

    if norm:
        return normalize(score, a)
    else:
        return score


@cython.ccall
@cython.locals(
    state=cython.object,
    player=cython.int,
    m_score=cython.double,
    m_seed_diff=cython.double,
    m_empty=cython.double,
    m_double=cython.double,
    m_capture=cython.double,
    m_opp_disc=cython.double,
    a=cython.int,
    norm=cython.bint,
    player_store=cython.int,
    opponent_store=cython.int,
    player_seeds=cython.int,
    opponent_seeds=cython.int,
    player_double_moves=cython.int,
    opponent_double_moves=cython.int,
    player_capture_moves=cython.int,
    opponent_capture_moves=cython.int,
    empty_opponent_houses=cython.int,
    empty_player_houses=cython.int,
    i=cython.int,
    seeds=cython.int,
    score=cython.double,
    evaluation=cython.double,
)
# Results from genetic optimization:
# https://docs.google.com/spreadsheets/d/1ubiIoIQEVj7INwhGG64uXSe--tWFOF7096EQ2GT8p2s/edit#gid=0
def evaluate_kalah_enhanced(
    state,
    player,
    m_score=18.0,
    m_seed_diff=0.1,
    m_empty=6.5,
    m_double=8.7,
    m_capture=6.5,
    m_opp_disc=0.75,
    a=5,
    norm=False,
):
    player_store = 6 if player == 1 else 13
    opponent_store = 6 if player == 2 else 13

    player_seeds = opponent_seeds = 0
    player_double_moves = opponent_double_moves = 0
    player_capture_moves = opponent_capture_moves = 0
    empty_opponent_houses = empty_player_houses = 0
    board: cython.list = state.board

    for i in range(0, 13):  # go through all houses (excluding p2 store)
        if i == 6 or i == 13:  # Skip p1/p2 store
            continue

        seeds = board[i]
        # Check if the house belongs to the player or the opponent
        if (player == 1 and 0 <= i <= 5) or (player == 2 and 7 <= i <= 12):
            player_seeds += seeds

            if seeds == 0:
                empty_player_houses += 1
            else:
                if m_double != 0 and calc_last_index_total_steps(seeds, i, opponent_store)[0] == player_store:
                    player_double_moves += 1
                elif m_capture != 0 and is_capture(board, i, player):
                    player_capture_moves += 1

        elif (player == 2 and 0 <= i <= 5) or (player == 1 and 7 <= i <= 12):
            opponent_seeds += seeds
            if seeds == 0:
                empty_opponent_houses += 1
            else:
                if m_double != 0 and calc_last_index_total_steps(seeds, i, player_store)[0] == opponent_store:
                    opponent_double_moves += 1
                elif m_capture != 0 and is_capture(board, i, 3 - player):
                    opponent_capture_moves += 1

    evaluation = (
        (m_score * (board[player_store] - board[opponent_store]))
        + (m_seed_diff * (player_seeds - opponent_seeds))
        + (m_empty * (empty_opponent_houses - empty_player_houses))
        + (m_double * (player_double_moves - opponent_double_moves))
        + (m_capture * (player_capture_moves - opponent_capture_moves))
    )

    if state.player == 3 - player:
        evaluation *= m_opp_disc

    if norm:
        return normalize(evaluation, a)
    else:
        return evaluation


@cython.ccall
@cython.locals(
    board=cython.list,
    move=cython.int,
    player=cython.int,
    seeds=cython.int,
    size=cython.int,
    last_index=cython.int,
    total_steps=cython.int,
    opp=cython.int,
    passed_opp=cython.bint,
    rounds=cython.int,
)
def is_capture(board, move, player):
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

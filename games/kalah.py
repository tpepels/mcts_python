# cython: language_level=3
# distutils: language=c++

import array
import cython
import numpy as np
from cython.cimports.libcpp.vector import vector
from cython.cimports.includes import GameState, win, loss, draw, normalize, c_random

import random

MAX_SEEDS = 280  # An approximation of the maximum number of seeds in a single position

# TODO 6x6 oplossen
# TODO Solved positions printen


@cython.cclass
class KalahGameState(GameState):
    players_bitstrings = [random.randint(1, 2**60 - 1) for _ in range(3)]

    zobrist_tables = {
        size: [[random.randint(1, 2**60 - 1) for _ in range(MAX_SEEDS)] for _ in range(size)]
        for size in range(14, 19)  # Assuming anything between a 6 and 8 pit board
    }

    zobrist_table = [[random.randint(1, 2**60 - 1) for _ in range(MAX_SEEDS)] for _ in range(14)]
    REUSE = False

    # player = cython.declare(cython.int, visibility="public")
    board = cython.declare(cython.int[:], visibility="public")
    board_hash = cython.declare(cython.longlong, visibility="public")
    last_action = cython.declare(cython.tuple[cython.int], visibility="public")

    num_houses: cython.int
    winner: cython.int
    p1_pit: cython.int
    p2_pit: cython.int

    n_moves: cython.int
    pie_rule_decision_made: cython.bint

    def __init__(self, board=None, player=1, last_action=None, winner=0, init_seeds=4, n_houses=6, n_moves=0):
        """
        Initialize the Kalah game state.

        :param board: Optional board state, represented as a list of integers (default: None).
        :param player: The player whose turn it is, 1 for player 1 and 2 for player 2 (default: 1).
        """
        self.player = player
        self.num_houses = n_houses
        self.winner = winner
        self.n_moves = n_moves

        self.p1_pit = n_houses
        self.p2_pit = (n_houses * 2) + 1

        self.pie_rule_decision_made = 0
        if n_moves == 1 and last_action == (-2,):
            self.pie_rule_decision_made = 1

        if board is None:
            self.board = np.full((n_houses * 2) + 2, init_seeds, dtype=np.int32)  # n houses plus two pits
            self.board[self.p1_pit] = 0  # player 1's pit
            self.board[self.p2_pit] = 0  # player 2's pit
            self.last_action = (-1,)
        else:
            self.board = board
            self.last_action = last_action

        # Recalculate the hash after every move
        zobrist_table = self.zobrist_tables[self.board.shape[0]]

        self.board_hash = 0
        position: cython.int
        for position in range((n_houses * 2) + 2):
            seeds = self.board[position]
            self.board_hash ^= zobrist_table[position][seeds]

        # XOR with the bitstring of the current player
        self.board_hash ^= self.players_bitstrings[self.player]

    @cython.cfunc
    @cython.locals(next_player=cython.int, winner=cython.int)
    def apply_action_playout(self, action: cython.tuple) -> cython.void:
        next_player, winner = self._apply_action_logic(action[0], self.board)
        # # The first two moves, no doubles
        # if self.n_moves < 2:  # This is the reason for resetting the number of moves above
        #     next_player = 3 - self.player

        self.winner = winner
        self.player = next_player
        self.last_action = action
        self.n_moves += 1
        # * Note that the hash is not recomputed here, as it is not needed for playouts

    @cython.ccall
    @cython.locals(next_player=cython.int, winner=cython.int)
    def apply_action(self, action: cython.tuple) -> KalahGameState:
        """
        Apply an action to the current game state and return the resulting new state.
        The state of the instance is not altered in this method.

        :param action: The action to apply, represented as the index of the house to pick up seeds from.
        :return: The resulting new game state after applying the action.
        """
        action_i: cython.int = action[0]
        new_board: cython.int[:] = self.board.copy()
        move_increment: cython.int = 1
        if action_i == -2:
            self.apply_pie_rule(new_board)
            next_player = 2
            winner = 0
            move_increment = 0
        else:
            next_player, winner = self._apply_action_logic(action_i, new_board)

            # # The first two moves, no doubles
            # if self.n_moves < 2:  # This is the reason for resetting the number of moves above
            #     next_player = 3 - self.player

        return KalahGameState(
            board=new_board,
            player=next_player,
            winner=winner,
            last_action=action,
            n_moves=self.n_moves + move_increment,
            n_houses=self.num_houses,
        )

    @cython.cfunc
    @cython.infer_types(True)
    def apply_pie_rule(self, board: cython.int[:]):
        assert self.n_moves == 1, "Pie Rule decision must be made after the first move!"
        assert not self.pie_rule_decision_made, "Pie Rule decision has already been made!"

        # Store pits temporarily
        temp_p1_pit = board[self.p1_pit]
        temp_p2_pit = board[self.p2_pit]

        # Reverse the houses
        for i in range(self.num_houses):
            board[i], board[self.num_houses + 1 + i] = board[self.num_houses + 1 + i], board[i]

        # Restore the pits
        board[self.p1_pit] = temp_p2_pit
        board[self.p2_pit] = temp_p1_pit

    @cython.cfunc
    @cython.locals(
        next_player=cython.int,
        last_index=cython.int,
        captured_seeds=cython.int,
        opposite_house=cython.int,
        winner=cython.int,
    )
    def _apply_action_logic(
        self, action_i: cython.int, board: cython.int[:]
    ) -> cython.tuple[cython.int, cython.int]:
        assert self.winner == 0
        assert board[action_i] > 0
        assert (self.player == 1 and 0 <= action_i < self.p1_pit) or (
            self.player == 2 and self.p1_pit < action_i <= self.p2_pit
        )

        seeds: cython.int = board[action_i]
        board[action_i] = 0

        next_player: cython.int = self.player
        last_index: cython.int = action_i

        while seeds > 0:
            last_index = (last_index + 1) % board.shape[0]

            # Skip the opponent's store
            if (self.player == 1 and last_index == self.p2_pit) or (
                self.player == 2 and last_index == self.p1_pit
            ):
                continue

            board[last_index] += 1
            seeds -= 1

            # If the last seed is sown in the player's store, it's their turn again
            if seeds == 0 and (
                (self.player == 1 and last_index == self.p1_pit)
                or (self.player == 2 and last_index == self.p2_pit)
            ):
                next_player: cython.int = self.player
            else:
                next_player: cython.int = 3 - self.player

        # Capture opponent's seeds if the last seed is sown in an empty house on the player's side and the opposite house has seeds.
        if (
            (self.player == 1 and 0 <= last_index < self.p1_pit)
            or (self.player == 2 and self.num_houses < last_index < self.p2_pit)
            and board[last_index] == 1
        ):
            opposite_house: cython.int = (self.board.shape[0] - 2) - last_index
            if board[opposite_house] > 0:
                # Capture seeds
                captured_seeds: cython.int = (
                    board[opposite_house] + 1
                )  # The one seed left in the final position is also captured
                board[opposite_house] = 0
                board[last_index] = 0

                # Add captured seeds to the player's store
                if self.player == 1:
                    board[self.p1_pit] += captured_seeds
                else:
                    board[self.p2_pit] += captured_seeds

        p1_houses_empty: cython.bint = True
        p2_houses_empty: cython.bint = True

        i: cython.int
        for i in range(0, self.p1_pit):
            if board[i] != 0:
                p1_houses_empty = False
                break

        for i in range(self.p1_pit + 1, self.p2_pit):
            if board[i] != 0:
                p2_houses_empty = False
                break

        winner: cython.int = 0
        if p1_houses_empty or p2_houses_empty:
            if p1_houses_empty:
                for i in range(self.p1_pit + 1, self.p2_pit):
                    board[self.p2_pit] += board[i]
                    board[i] = 0
            else:
                for i in range(0, self.p1_pit):
                    board[self.p1_pit] += board[i]
                    board[i] = 0

            # If one of the players has no legal moves remaining, the game is terminal
            if board[self.p1_pit] > board[self.p2_pit]:
                winner = 1
            elif board[self.p1_pit] < board[self.p2_pit]:
                winner = 2
            else:
                winner = -1  # For a terminal state that is a draw.

        return next_player, winner

    @cython.cfunc
    def skip_turn(self) -> KalahGameState:
        """Used for the null-move heuristic in alpha-beta search

        Returns:
            BreakthroughGameState: A new gamestate in which the players are switched but no move performed
        """
        return KalahGameState(
            self.board.copy(),
            3 - self.player,
            winner=self.winner,
            last_action=self.last_action,
            n_houses=self.num_houses,
        )

    @cython.cfunc
    def get_random_action(self) -> cython.tuple:
        p_store: cython.int = self.p1_pit if self.player == 1 else self.p2_pit
        opp_store: cython.int = self.p1_pit if self.player == 2 else self.p2_pit

        start_h: cython.int = 0 if self.player == 1 else self.num_houses + 1
        start_i: cython.int = c_random(0, self.num_houses - 1)

        caps: cython.list = []
        doubles: cython.list = []
        moves: cython.list = []

        i: cython.int
        for i in range(self.num_houses):
            action_i: cython.int = start_h + ((start_i + i) % self.num_houses)

            if self.board[action_i] > 0:
                action: cython.tuple[cython.int] = (action_i,)
                # Captures
                if self.is_capture(action):
                    caps.append(action)
                # Double moves
                elif (
                    caps == []
                    and steps_index(
                        seeds=self.board[action_i], move=action_i, k=opp_store, size=self.board.shape[0]
                    )[0]
                    == p_store
                ):
                    doubles.append(action)
                # Regular moves
                elif caps == [] and doubles == []:
                    moves.append(action)

        # * Make captures whenever you can, otherwise double moves, otherwise regular moves
        if caps != []:
            return caps[c_random(0, len(caps) - 1)]
        elif doubles != []:
            return doubles[c_random(0, len(doubles) - 1)]
        elif moves != []:
            return moves[c_random(0, len(moves) - 1)]

        assert False, "No legal actions found!"

    @cython.ccall
    @cython.locals(i=cython.int)
    def get_legal_actions(self) -> cython.list:
        """
        Get a list of legal actions for the current game state.

        :return: A list of legal actions, represented as the indices of the houses with seeds.
        """
        start: cython.int = 0 if self.player == 1 else self.num_houses + 1
        result: cython.list = []
        for i in range(start, start + self.num_houses):
            if self.board[i] > 0:
                result.append((i,))

        if self.n_moves == 1 and not self.pie_rule_decision_made:
            result.append((-2,))

        return result

    @cython.ccall
    def is_terminal(self) -> cython.bint:
        """
        Check if the current game state is a terminal state, i.e., the game has ended.

        :return: True if the state is terminal, False otherwise.
        """
        return self.winner != 0

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def get_reward(self, player: cython.int) -> cython.int:
        """
        Returns the reward value of a terminal state [-1, 0, 1] (loss/draw/win)

        :return: The reward value.
        """
        if self.winner == 0:
            return 0

        if self.winner == -1:
            return draw
        elif (player == 1 and self.winner == 1) or (player == 2 and self.winner == 2):
            return win
        else:
            return loss

    @cython.ccall
    def get_result_tuple(self) -> cython.tuple:
        if self.winner == 1:
            return (1.0, 0.0)
        elif self.winner == 2:
            return (0.0, 1.0)

        return (0.5, 0.5)

    @cython.cfunc
    @cython.locals(last_index=cython.int, total_steps=cython.int)
    def is_capture(self, move: cython.tuple) -> cython.bint:
        """
        Check if the given move results in a capture.

        :param move: The move to check, represented as the index of the house to pick up seeds from.
        :return: True if the move results in a capture, False otherwise.
        """
        move_i: cython.int = move[0]
        seeds: cython.int = self.board[move_i]
        # Calculate how often the opponent's pit is passed
        size: cython.int = self.board.shape[0]
        last_index, total_steps = steps_index(
            seeds=seeds,
            move=move_i,
            k=self.p1_pit if self.player == 2 else self.p2_pit,
            size=size,
        )
        opp: cython.int = (size - 2) - last_index
        passed_opp: cython.int = total_steps > opp if move_i < opp else total_steps > size + opp
        rounds: cython.int = seeds // (size - 1)

        # If we end up where we started, that means that we are in a house with 0 seeds, but the board still
        # has the seeds in it.
        if ((self.board[last_index] == 0 and rounds == 0) or last_index == move_i) and (
            (self.player == 1 and 0 <= last_index < self.p1_pit)
            or (self.player == 2 and self.p1_pit < last_index < self.p2_pit)
        ):
            # Either the opposing side had a stone before, or we dropped one in it
            if self.board[opp] > 0 or passed_opp:
                return 1

        return 0

    @cython.cfunc
    def evaluate_moves(self, moves: cython.list) -> cython.list:
        """
        Evaluate the given moves using a heuristic based on the potential benefits of the move.

        :param moves: The list of moves to evaluate.
        :return: The list of heuristic scores for the moves.
        """
        scores: cython.list = [()] * len(moves)
        for i in range(len(moves)):
            scores[i] = (moves[i], self.evaluate_move(moves[i]))
        return scores

    @cython.cfunc
    def move_weights(self, moves: cython.list) -> cython.list:
        """
        Evaluate the given moves using a heuristic based on the potential benefits of the move.

        :param moves: The list of moves to evaluate.
        :return: The list of heuristic scores for the moves.
        """
        n_moves = len(moves)
        scores: vector[cython.int]
        scores.reserve(n_moves)
        i: cython.int
        for i in range(n_moves):
            move: cython.tuple = moves[i]
            scores.push_back(self.evaluate_move(move))
        return scores

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    @cython.locals(
        move_i=cython.int, score=cython.int, last_index=cython.int, board=cython.list, player=cython.int
    )
    def evaluate_move(self, move: cython.tuple) -> cython.int:
        """
        Evaluates the given move using a heuristic based on the potential benefits of the move.

        :param move: The move to evaluate.
        :return: The heuristic score for the move.
        """
        move_i: cython.int = move[0]
        # Initialize score to the number of seeds in the house
        score = self.board[move_i]
        size: cython.int = self.board.shape[0]
        # Check if the move results in a capture and assign a positive score
        if self.is_capture(move):
            last_index, _ = steps_index(
                seeds=self.board[move_i],
                move=move_i,
                k=self.p1_pit if self.player == 2 else self.p2_pit,
                size=self.board.shape[0],
            )
            score += self.board[(size - 2) - last_index] * 4
        # Check if the move results in another move for the current player and assign a positive score
        last_index, _ = steps_index(
            seeds=self.board[move_i],
            move=move_i,
            k=self.p2_pit if self.player == 1 else self.p1_pit,
            size=self.board.shape[0],
        )
        if (self.player == 1 and last_index == self.p1_pit) or (
            self.player == 2 and last_index == self.p1_pit
        ):
            score += 10

        return score

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
        player2_store = str(self.board[self.p2_pit]).rjust(2)
        player1_store = str(self.board[self.num_houses]).rjust(2)
        output = (
            f"Player 1's store: {player1_store}\n"
            f"{top_row}\n"
            f"{bottom_row}\n"
            f"Player 2's store: {player2_store}\n"
        )
        output += "..." * 20 + "\n"

        if full_debug:
            actions = self.get_legal_actions()
            output += f"Player: {self.player} | {len(actions)} actions: {[a[0] for a in actions]} | hash: {self.board_hash}\n"
            output += f"Reward: {self.get_reward(1)}/{self.get_reward(2)} | Terminal?: {self.is_terminal()}\n"
            # ------------------ Evaluation ----------------------
            output += "-*-" * 8 + "\n"
            enhanced_eval_1 = self.evaluate(1, self.default_params, False)
            enhanced_eval_2 = self.evaluate(2, self.default_params, False)
            enhanced_eval_1_norm = self.evaluate(1, self.default_params, True)
            enhanced_eval_2_norm = self.evaluate(2, self.default_params, True)
            output += f"Eval P1: {enhanced_eval_1:.2f} P2: {enhanced_eval_2:.2f} | "
            output += f"normalized P1: {enhanced_eval_1_norm:.2f} P2: {enhanced_eval_2_norm:.2f}\n"
            output += "-*-" * 8 + "\n"

            for action in actions:
                output += f"{action} is capture? {self.is_capture(action)}\n"

            if len(actions) > 0:
                actions = self.evaluate_moves(self.get_legal_actions())
                # actions = sorted(actions, key=lambda x: x[1], reverse=True)
                output += "..." * 20 + "\n"
                output += str(actions)

        return output

    param_order: dict = {
        "m_score": 0,
        "m_seed_diff": 1,
        "m_empty": 2,
        "m_double": 3,
        "m_capture": 4,
        "m_opp_disc": 5,
        "a": 6,
    }

    default_params = array.array("d", [18.0, 0.1, 6.5, 8.7, 6.5, 0.95, 200])

    @cython.cfunc
    @cython.exceptval(-9999999, check=False)
    def evaluate(
        self, player: cython.int, params: cython.double[:], norm: cython.bint = False
    ) -> cython.double:
        p_store: cython.int = self.p1_pit if player == 1 else self.p2_pit
        opp_store: cython.int = self.p1_pit if player == 2 else self.p2_pit

        seed_diff: cython.int = 0
        double_move_diff: cython.int = 0
        capture_diff: cython.int = 0
        empty_house_diff: cython.int = 0

        # "m_score": 0,
        # "m_seed_diff": 1,
        # "m_empty": 2,
        # "m_double": 3,
        # "m_capture": 4,
        # "m_opp_disc": 5,
        # "a": 6,

        i: cython.int
        for i in range(0, self.p2_pit):  # go through all houses (excluding p2 store)
            if i == self.p1_pit or i == self.p2_pit:  # Skip p1/p2 store
                continue

            seeds: cython.int = self.board[i]
            # Check if the house belongs to the player or the opponent
            if (player == 1 and 0 <= i < self.p1_pit) or (player == 2 and self.p1_pit < i < self.p2_pit):
                mult: cython.int = 1
                # * These are needed to flip the viewpoint based on the player under consideration
                m_store: cython.int = p_store
                o_store: cython.int = opp_store
            else:
                mult: cython.int = -1
                m_store: cython.int = opp_store
                o_store: cython.int = p_store

            seed_diff += mult * seeds

            if seeds == 0:
                empty_house_diff += mult
            else:
                if (
                    seeds > 0
                    and params[3] != 0
                    and steps_index(seeds=seeds, move=i, k=o_store, size=self.board.shape[0])[0] == m_store
                ):
                    double_move_diff += mult * 1
                if (
                    seeds > 0
                    and params[4] != 0
                    and is_capture(
                        board=self.board,
                        move=i,
                        player=player if mult == 1 else 3 - player,
                        p1_pit=self.p1_pit,
                        p2_pit=self.p2_pit,
                    )
                    == 0
                ):
                    capture_diff += mult * 1

        evaluation = (
            (params[0] * (self.board[p_store] - self.board[opp_store]))
            + (params[1] * seed_diff)
            + (params[2] * empty_house_diff)
            + (params[3] * double_move_diff)
            + (params[4] * capture_diff)
        )

        if self.player == 3 - player:
            evaluation *= params[5]

        if norm:
            return normalize(evaluation, params[6])
        else:
            return evaluation

    @property
    def transposition_table_size(self):
        # return an appropriate size based on the game characteristics
        return 2**17

    def __repr__(self) -> str:
        return "kalah"


@cython.cfunc
@cython.inline
@cython.locals(
    seeds=cython.int,
    move=cython.int,
    k=cython.int,
    size=cython.int,
    passes=cython.int,
    remaining_steps=cython.int,
    total_steps=cython.int,
)
def steps_index(seeds, move, k, size) -> cython.tuple[cython.int, cython.int]:
    passes = seeds // size
    remaining_steps = seeds % size

    if remaining_steps + move >= size + k:
        passes += 1

    # Calculate total steps considering the opponent's store
    total_steps = move + seeds + passes
    return total_steps % ((size - 1) + passes), total_steps


@cython.ccall
@cython.locals(
    board=cython.int[:],
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
def is_capture(board, move, player, p1_pit, p2_pit) -> cython.bint:
    seeds = board[move]

    # Calculate how often the opponent's pit is passed
    size = board.shape[0]
    last_index, total_steps = steps_index(
        seeds=seeds,
        move=move,
        k=p1_pit if player == 2 else p2_pit,
        size=size,
    )
    opp = (board.shape[0] - 2) - last_index
    passed_opp = total_steps > opp if move < opp else total_steps > size + opp
    rounds = seeds // (size - 1)

    # If we end up where we started, that means that we are in a house with 0 seeds, but the board still
    # has the seeds in it.
    if ((board[last_index] == 0 and rounds == 0) or last_index == move) and (
        (player == 1 and 0 <= last_index < p1_pit) or (player == 2 and p1_pit < last_index < p2_pit)
    ):
        # Either the opposing side had a stone before, or we dropped one in it
        if board[opp] > 0 or passed_opp:
            return 1

    return 0

from pprint import pprint
import random
import time
from ai.ai_player import AIPlayer


class SearchTimeout(Exception):
    """Exception raised when the search times out."""

    pass


class AlphaBetaPlayer(AIPlayer):
    def __init__(
        self, player, max_depth, max_time, evaluate, use_null_moves=False, use_quiescence=False, debug=False
    ):
        # Identify the player
        self.player = player
        # Set the maximum depth for the iterative deepening search
        self.max_depth = max_depth
        # The evaluation function to be used
        self.evaluate = evaluate
        # Whether or not to use the null-move heuristic
        self.use_null_moves = use_null_moves
        # Whether or not to use quiescence search
        self.use_quiescence = use_quiescence
        # Depth reduction for the null-move heuristic
        self.R = 2
        # Maximum time for a move in seconds
        self.max_time = max_time
        # The depth limit beyond which the search should always finish
        self.interrupt_depth_limit = 2
        # Whether to print debug statistics
        self.debug = debug
        self.statistics = []

    def best_action(self, state):
        # Reset debug statistics
        self.quiescence_searches = nodes_visited = cutoffs = 0
        max_depth_reached = null_moves_cutoff = total_moves_generated = 0
        search_times = []

        def value(state, alpha, beta, depth, null=True):
            nonlocal nodes_visited, cutoffs, max_depth_reached, null_moves_cutoff, search_times, total_moves_generated

            nodes_visited += 1
            if (depth <= self.interrupt_depth_limit) and (time.time() - start_time > (self.max_time + 10)):
                raise SearchTimeout()

            max_player = state.player == self.player

            # If the state is terminal, return the reward
            if state.is_terminal():
                return state.get_reward(self.player), None

            # If maximum depth is reached, return the evaluation
            if depth == 0:
                if self.use_quiescence:
                    return self.quiescence(state, alpha, beta), None
                else:
                    return self.evaluate(state, self.player), None

            if max_player:
                # If using null-move heuristic, attempt a null move
                if self.use_null_moves and null and depth >= self.R + 1:
                    null_state = state.skip_turn()
                    null_score, _ = value(null_state, alpha, beta, depth - self.R - 1, False)
                    if null_score >= beta:
                        null_moves_cutoff += 1
                        cutoffs += 1  # Add this line
                        # print(f"null {null_score=} >= {beta=}")
                        return null_score, None

                # Max player's turn
                v = -float("inf")
                best_move = None
                actions = state.get_legal_actions()
                total_moves_generated += len(actions)
                # Order moves by their heuristic value
                actions.sort(
                    key=lambda move: -state.evaluate_move(move)
                    if self.player == state.player
                    else state.evaluate_move(move)
                )

                for move in actions:
                    new_state = state.apply_action(move)
                    min_v, _ = value(new_state, alpha, beta, depth - 1)
                    if min_v > v:
                        v = min_v
                        best_move = move
                    if v >= beta:
                        cutoffs += 1  # Add this line
                        return v, move
                    alpha = max(alpha, v)

                if best_move is None:
                    return v, random.sample(actions, 1)[0]

                return v, best_move

            else:  # minimizing player
                v = float("inf")
                actions = state.get_legal_actions()
                total_moves_generated += len(actions)
                # Order moves by heuristic value
                actions.sort(
                    key=lambda move: state.evaluate_move(move)
                    if self.player == state.player
                    else -state.evaluate_move(move)
                )
                for move in actions:
                    new_state = state.apply_action(move)
                    max_v, _ = value(new_state, alpha, beta, depth - 1)
                    if max_v < v:
                        cutoffs += 1  # Add this line
                        v = max_v
                    if v <= alpha:
                        return v, None
                    beta = min(beta, v)
                return v, None

        start_time = time.time()
        v, best_move = None, None
        try:
            for depth in range(1, self.max_depth + 1):
                if time.time() - start_time >= self.max_time:
                    break  # Stop searching if the time limit has been exceeded
                v, best_move = value(state, -float("inf"), float("inf"), depth, True)
                search_times.append(time.time() - start_time)
                max_depth_reached = depth
            # Search can stop if it runs overtime, so make sure to use the correct return values
        except SearchTimeout:
            pass  # If search times out, return the best move found so far

        if self.debug:
            stat_dict = {
                "max_player": self.player,
                "nodes_visited": nodes_visited,
                "cutoffs": cutoffs,
                "null_moves_cutoff": null_moves_cutoff,
                "max_depth_reached": max_depth_reached,
                "quiescence_searches": self.quiescence_searches,
                "total_search_time": time.time() - start_time,
                "search_times_per_level": search_times,
                "average_branching_factor": total_moves_generated / nodes_visited,
                "moves_generated": total_moves_generated,
            }
            self.statistics.append(stat_dict)
            pprint(stat_dict, compact=True)

        return best_move, v

    def quiescence(self, state, alpha, beta):
        # Increment the number of quiescence searches
        self.quiescence_searches += 1
        # Quiescence search to avoid the horizon effect
        stand_pat = self.evaluate(state, self.player)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        for move in state.get_legal_actions():
            if state.is_capture(move):
                new_state = state.apply_move(move)
                score = -self.quiescence(new_state, -beta, -alpha)
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
        return alpha

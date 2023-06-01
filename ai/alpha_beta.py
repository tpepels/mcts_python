import math
from pprint import pprint
import random
import time
from ai.ai_player import AIPlayer
from ai.transpos_table import TranspositionTable


class AlphaBetaPlayer(AIPlayer):
    def __init__(
        self,
        player,
        max_depth,
        max_time,
        evaluate,
        transposition_table_size=2**16,
        use_null_moves=False,
        use_quiescence=False,
        debug=False,
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
        # Store evaluation resuls
        self.trans_table = TranspositionTable(transposition_table_size)

    def best_action(self, state):
        # Reset debug statistics
        self.quiescence_searches = nodes_visited = cutoffs = evaluated = transpos = 0
        max_depth_reached = null_moves_cutoff = total_moves_generated = 0
        # Keep track of search times per level
        search_times = []

        def value(state, alpha, beta, depth, allow_null_move=True, root=False):
            nonlocal evaluated, nodes_visited, cutoffs, max_depth_reached, transpos
            nonlocal total_moves_generated, null_moves_cutoff, search_times

            null_move_result = not allow_null_move
            # null is true if a null-move is possible. If a null move cannot be made then, we are in a part of
            # the tree where a null move was previously made. Hence we cannot use transpositions.
            # i.e. do not consider illegal moves in the hash-table
            if allow_null_move and not root:
                if not self.debug:
                    v = self.trans_table.get(state.board_hash, depth, state.player)
                else:
                    v = self.trans_table.get(state.board_hash, depth, state.player, board=state.board)

                if v is not None:
                    return v, None

            try:
                is_max_player = state.player == self.player

                if state.is_terminal():
                    v = state.get_reward(self.player)
                    return v, None

                if depth == 0:
                    evaluated += 1
                    if self.use_quiescence:
                        v = self.quiescence(state, alpha, beta)
                    else:
                        v = self.evaluate(state, self.player)

                    return v, None

                if is_max_player:
                    # Apply a null move to check if skipping turns results in better results than making a move
                    if self.use_null_moves and allow_null_move and depth >= self.R + 1:
                        null_state = state.skip_turn()
                        null_score, _ = value(null_state, alpha, beta, depth - self.R - 1, False)
                        null_move_result = True  # Set flag to True so result is not stored
                        if null_score >= beta:
                            null_moves_cutoff += 1
                            cutoffs += 1
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
                    nodes_visited += 1
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
                        return v, random.choice(actions)

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
                    nodes_visited += 1
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

            finally:
                if not null_move_result:  # If not a null-move result
                    self.trans_table.put(
                        state.board_hash,
                        v,
                        depth,
                        state.player,
                        board=state.board if self.debug else None,
                    )

        start_time = time.time()
        v, best_move = None, None
        last_time = 0

        for depth in range(1, self.max_depth + 1):
            if (time.time() - start_time) + last_time >= self.max_time:
                break  # Stop searching if the time limit has been exceeded or if there's not enough time to do another search

            v, best_move = value(state, -float("inf"), float("inf"), depth, True, root=True)
            max_depth_reached = depth

            # keep track of the time spent
            last_time = time.time() - start_time
            search_times.append(last_time)

        if self.debug:
            stat_dict = {
                "max_player": self.player,
                "nodes_visited": nodes_visited,
                "nodes_evaluated": evaluated,
                "cutoffs": cutoffs,
                "max_depth_reached": max_depth_reached,
                "search_times_per_level": search_times,
                "average_branching_factor": int(round(evaluated / nodes_visited, 0)),
                "moves_generated": total_moves_generated,
            }

            if self.use_quiescence:
                stat_dict["quiescence_searches"] = self.quiescence_searches
            if self.use_null_moves:
                stat_dict["null_moves_cutoff"] = null_moves_cutoff

            self.statistics.append(stat_dict)
            pprint(stat_dict, compact=True)
            pprint(self.trans_table.get_metrics(), compact=True)
            self.trans_table.reset_metrics()

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

import math
from pprint import pprint
import random
import time
from ai.ai_player import AIPlayer
from ai.transpos_table import TranspositionTable
from games.gamestate import GameState

total_search_time = {1: 0, 2: 0}
n_moves = {1: 0, 2: 0}
depth_reached = {1: 0, 2: 0}


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
        use_transpositions=True,
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
        if use_transpositions:
            self.trans_table = TranspositionTable(transposition_table_size)
        else:
            self.trans_table = None

        global total_search_time, n_moves, depth_reached
        total_search_time[player] = 0
        n_moves[player] = 0
        depth_reached[player] = 0

    def best_action(self, state: GameState):
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
            if allow_null_move and not root and self.trans_table is not None:
                # if not self.debug:
                v = self.trans_table.get(state.board_hash, depth, state.player)
                # else:
                #     v = self.trans_table.get(state.board_hash, depth, state.player, board=state.board)

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
                        if len(actions) == 0:
                            print(state.visualize())
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
                if not null_move_result and self.trans_table is not None:  # If not a null-move result
                    self.trans_table.put(
                        state.board_hash,
                        v,
                        depth,
                        state.player,
                        # board=state.board if self.debug else None,
                    )

        start_time = time.time()
        v, best_move = None, None

        empirical_factor = 1
        search_times = [0]  # Initialize with 0 for depth 0
        for depth in range(1, self.max_depth + 1):  # Start depth from 1
            if depth > 2:
                if search_times[-2] == 0:
                    empirical_factor = 1
                else:
                    # Compute the empirical factor as the ratio of the last two search times
                    empirical_factor = search_times[-1] / search_times[-2]
                if (time.time() - start_time) + (empirical_factor * search_times[-1]) >= self.max_time:
                    break  # Stop searching if the time limit has been exceeded or if there's not enough time to do another search

            if self.debug:
                print(
                    f"Searching depth: {depth} with {self.max_time - (time.time() - start_time)} seconds left. {empirical_factor=}"
                )

            start_depth_time = time.time()  # Time when this depth search starts
            v, best_move = value(state, -float("inf"), float("inf"), depth, True, root=True)
            max_depth_reached = depth
            # keep track of the time spent
            depth_time = time.time() - start_depth_time  # Time spent on this depth
            search_times.append(depth_time)

        if self.debug:
            global total_search_time, n_moves

            total_search_time[self.player] += time.time() - start_time
            n_moves[self.player] += 1
            depth_reached[self.player] += max_depth_reached

            stat_dict = {
                "max_player": self.player,
                "nodes_visited": nodes_visited,
                "nodes_evaluated": evaluated,
                "cutoffs": cutoffs,
                "max_depth_reached": max_depth_reached,
                "search_times_per_level": search_times,
                "average_branching_factor": int(round(evaluated / nodes_visited, 0)),
                "moves_generated": total_moves_generated,
                "average_search_time": int(total_search_time[self.player] / n_moves[self.player]),
                "average_depth_reached": int(depth_reached[self.player] / n_moves[self.player]),
            }

            if self.use_quiescence:
                stat_dict["quiescence_searches"] = self.quiescence_searches
            if self.use_null_moves:
                stat_dict["null_moves_cutoff"] = null_moves_cutoff

            self.statistics.append(stat_dict)
            if self.trans_table is not None:
                pprint({**stat_dict, **self.trans_table.get_metrics()}, compact=True)
                self.trans_table.reset_metrics()
            else:
                pprint(stat_dict, compact=True)

        return best_move, v

    def quiescence(self, state: GameState, alpha, beta):
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

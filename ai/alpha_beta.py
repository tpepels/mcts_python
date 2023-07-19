import time


from ai.ai_player import AIPlayer
from ai.transpos_table import TranspositionTable
from games.gamestate import GameState, win, loss, draw
from util import pretty_print_dict

total_search_time = {1: 0, 2: 0}
n_moves = {1: 0, 2: 0}
depth_reached = {1: 0, 2: 0}
best_value_labels = {win: "WIN", draw: "DRAW", loss: "LOSS"}


class AlphaBetaPlayer(AIPlayer):
    cumulative_keys = [
        "total_null_cutoffs",
        "total_q_searches",
        "total_nodes_visited",
        "total_nodes_evaluated",
        "total_nodes_cutoff",
        "total_nodes_generated",
        "total_depth_reached",
        "max_depth_finished",
        "total_search_time",
        "count_searches",
        "count_tim_out",
    ]

    def __init__(
        self,
        player,
        max_time,
        evaluate,
        max_depth=25,
        transposition_table_size=2**16,
        use_null_moves=False,
        use_quiescence=False,
        use_tt=True,
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
        # instance-level statistics
        self.stats = []
        self.c_stats = {key: 0 for key in AlphaBetaPlayer.cumulative_keys}  # Initialize to 0
        self.trans_table = None
        if use_tt:  # Store evaluation resuls
            self.trans_table = TranspositionTable(transposition_table_size)
        global total_search_time, n_moves, depth_reached
        total_search_time[player] = 0
        n_moves[player] = 0
        depth_reached[player] = 0

    def best_action(self, state: GameState):
        assert state.player == self.player, "Player to move in the game is not my assigned player"
        # Reset debug statistics
        self.quiescence_searches = nodes_visited = cutoffs = evaluated = 0
        max_depth_reached = null_moves_cutoff = total_moves_generated = best_move_order = 0
        # Keep track of search times per level
        start_time = time.time()
        iteration_count = 0
        interrupted = False

        def value(
            state,
            alpha,
            beta,
            depth,
            max_d=0,
            allow_null_move=True,
            root=False,
        ):
            nonlocal evaluated, nodes_visited, cutoffs, start_time, interrupted
            nonlocal total_moves_generated, null_moves_cutoff, iteration_count, time_limit, best_move_order
            is_max_player = state.player == self.player

            if state.is_terminal():
                v = state.get_reward(self.player)  # evaluate in the view of the player to move
                return v, None

            if not interrupted:
                # Do this increment after the check or the timeout error will only occur once instead of at each level
                iteration_count += 1
            else:
                return 0, None

            # This function checks if we are running out of time.
            # Don't do this for the first depth, in some games, even the lowest search depth takes a lot of time..
            if not first_depth and iteration_count % 1000 == 0:  # Check every so many iterations
                if (time.time() - start_time) > time_limit:
                    interrupted = True
                    return 0, None

            if depth == 0 and allow_null_move and self.use_quiescence and not interrupted:
                v = self.quiescence(state, alpha, beta)

            # If we are interrupted, cut off the search and return evaluations straight away
            if depth == 0:
                evaluated += 1
                v = self.evaluate(state, self.player)  # evaluate in the view of the player to move
                return v, None

            # Apply a null move to check if skipping turns results in better results than making a move
            if self.use_null_moves and is_max_player and not root and allow_null_move and depth >= self.R + 1:
                null_state = state.skip_turn()
                # Perform a reduced-depth search
                null_score, _ = value(
                    state=null_state,
                    alpha=alpha,
                    beta=beta,
                    depth=(depth - self.R) - 1,
                    max_d=max_d,
                    allow_null_move=False,
                    root=False,
                )
                if null_score >= beta:
                    null_moves_cutoff += 1
                    cutoffs += 1
                    return null_score, None
            try:
                best_move = None
                v = float("-inf") if is_max_player else float("inf")

                # Check if a move already exists somewhere in the transposition table
                if self.trans_table:
                    _, best_move = self.trans_table.get(state.board_hash, 0, state.player, 0)

                actions = sorted(
                    state.evaluate_moves(state.get_legal_actions()), key=lambda x: x[1], reverse=is_max_player
                )
                actions = [move for move, _ in actions]
                total_moves_generated += len(actions)

                # If best_move exists, place it first in the list
                if best_move is not None and best_move in actions:
                    best_move_order += 1
                    actions.remove(best_move)
                    actions.insert(0, best_move)

                nodes_visited += 1

                for move in actions:
                    new_v, _ = value(
                        state=state.apply_action(move),
                        alpha=alpha,
                        beta=beta,
                        depth=depth - 1,
                        max_d=max_d,
                        root=False,
                        allow_null_move=allow_null_move,
                    )

                    # Update v, alpha or beta based on the player
                    if (is_max_player and new_v > v) or (not is_max_player and new_v < v):
                        v = new_v
                        best_move = move

                    if is_max_player:
                        alpha = max(alpha, new_v)
                    else:
                        beta = min(beta, new_v)

                    # Prune the branch
                    if beta <= alpha:
                        cutoffs += 1
                        break

                return v, best_move

            finally:
                # If not a null-move result
                if self.trans_table is not None and allow_null_move:
                    self.trans_table.put(
                        key=state.board_hash,
                        value=v,
                        depth=max_d - depth,
                        player=state.player,
                        best_move=best_move,
                        max_d=max_d,
                    )

        v, best_move = None, None
        search_times = [0.0]
        best_values = []
        last_best_move = None
        last_best_v = 0
        first_depth = True

        for depth in range(1, self.max_depth + 1):
            start_depth_time = time.time()  # Time when this depth search starts
            # How much time can be spent searching this depth
            time_limit = self.max_time - (start_depth_time - start_time)

            if interrupted or ((time.time() - start_time) + (search_times[-1]) >= self.max_time):
                # print(f"interrupted. {last_best_move=}, {last_best_v=}, {best_move=}, {v=}")
                break  # Stop searching if the time limit has been exceeded or if there's not enough time to do another search

            v, best_move = value(
                state,
                alpha=-float("inf"),
                beta=float("inf"),
                depth=depth,
                max_d=depth,
                allow_null_move=True,
                root=True,
            )
            # After the first search, we can interrupt the search
            first_depth = False
            if not interrupted:
                last_best_move = best_move
                last_best_v = v

                best_values.append((depth, v, best_move))
                max_depth_reached = depth
                if self.debug:
                    print(
                        f" d={depth} t_r={(time_limit):.2f} l_t={search_times[-1]:.2f} ***  ",
                        end="",
                    )

            # keep track of the time spent
            depth_time = time.time() - start_depth_time  # Time spent on this depth
            search_times.append(depth_time)
            # Don't spend any more time on proven wins/losses
            if v in [win, loss]:
                break

        if interrupted:
            best_move = last_best_move
            v = last_best_v

        if self.debug:
            print()
            global total_search_time, n_moves

            total_search_time[self.player] += time.time() - start_time
            n_moves[self.player] += 1
            depth_reached[self.player] += max_depth_reached

            stat_dict = {
                f"{self.player}_max_player": self.player,
                "nodes_best_move_order": best_move_order,
                "nodes_visited": nodes_visited,
                "nodes_evaluated": evaluated,
                "nodes_per_sec": nodes_visited / (time.time() - start_time),
                "nodes_cutoff": cutoffs,
                "nodes_avg_br_fact": int(round(total_moves_generated / nodes_visited, 0)),
                "nodes_generated": total_moves_generated,
                "depth_average": int(depth_reached[self.player] / n_moves[self.player]),
                "depth_finished": max_depth_reached,
                "depth": depth,
                "search_time": (time.time() - start_time),
                "search_times_p.d": search_times[1:],
                "search_time_average": int(total_search_time[self.player] / n_moves[self.player]),
                "search_tim_out": interrupted,
                "best_value": best_value_labels.get(v, v),
                "best_move": best_move,
                "best_values": best_values,
            }
            if v == win:
                stat_dict["best_value"] = "WIN"
            elif v == draw:
                stat_dict["best_value"] = "DRAW"
            elif v == loss:
                stat_dict["best_value"] = "LOSS"

            if self.use_quiescence:
                stat_dict["quiescence_searches"] = self.quiescence_searches
                self.c_stats["total_q_searches"] += self.quiescence_searches

            if self.use_null_moves:
                stat_dict["null_moves_cutoff"] = null_moves_cutoff
                self.c_stats["total_null_cutoffs"] += null_moves_cutoff

            self.stats.append(stat_dict)

            self.c_stats["total_nodes_visited"] += nodes_visited
            self.c_stats["total_nodes_evaluated"] += evaluated
            self.c_stats["total_nodes_cutoff"] += cutoffs
            self.c_stats["total_nodes_generated"] += total_moves_generated
            self.c_stats["total_depth_reached"] += max_depth_reached
            self.c_stats["max_depth_finished"] = max(max_depth_reached, self.c_stats["max_depth_finished"])
            self.c_stats["total_search_time"] += time.time() - start_time
            self.c_stats["count_searches"] += 1
            self.c_stats["count_tim_out"] += int(interrupted)
            if self.trans_table is not None:
                stat_dict = {**stat_dict, **self.trans_table.get_metrics()}
                self.trans_table.reset_metrics()
            pretty_print_dict(stat_dict)

        return best_move, v

    def quiescence(self, state: GameState, alpha, beta):
        # This is the quiescence function, which aims to mitigate the horizon effect by
        # conducting a more exhaustive search on volatile branches of the game tree,
        # such as those involving captures.

        stand_pat = self.evaluate(state, self.player)
        # Call the evaluation function to get a base score for the current game state.
        # This score is used as a baseline to compare with the potential outcomes of captures.

        if stand_pat >= beta:
            return beta
        # Beta represents the minimum guaranteed score that the maximizing player has. If our
        # stand_pat score is greater than or equal to beta, this branch is not going to be selected
        # by the maximizing player (since it already has a better or equal option), so we cut off
        # the search here and return beta.

        if alpha < stand_pat:
            alpha = stand_pat
        # Alpha represents the maximum score that the minimizing player is assured of. If our stand_pat
        # score is higher than alpha, it means that this branch could potentially provide a better
        # outcome for the minimizing player. Hence, we update alpha to this higher value.

        actions = state.get_legal_actions()
        for move in actions:
            # Iterate over all legal actions in the current game state.

            if state.is_capture(move):
                self.quiescence_searches += 1
                # Increment the counter that tracks the number of quiescence searches performed.

                # Check if the current move is a capture. We're specifically interested in captures
                # in a quiescence search, as they tend to create major swings in the game state.

                new_state = state.apply_action(move)
                # Apply the capture move to generate the new game state that results from this move.

                score = -self.quiescence(new_state, -beta, -alpha)
                # Perform a recursive quiescence search on the new game state to evaluate its score.
                # Note the negation of the score and the swapping of alpha and beta. This is a common
                # technique in adversarial search algorithms (like Alpha-Beta Pruning and Negamax),
                # representing the change in perspective between the minimizing and maximizing player.

                if score >= beta:
                    return beta
                # If the evaluated score is greater than or equal to beta, we return beta. This
                # represents a "beta-cutoff", indicating that the maximizing player has a better
                # or equal option elsewhere and wouldn't choose this branch.

                if score > alpha:
                    alpha = score
                # If the evaluated score is better than the current alpha, we update alpha.
                # This means that this branch might be a better option for the minimizing player.

        return alpha
        # At the end of the search, return alpha as the score for this position from the perspective
        # of the minimizing player. This score represents the best the minimizing player can achieve
        # in the worst case.

    def print_cumulative_statistics(self):
        if not self.debug:
            return

        # Compute the average at the end of the game(s):
        self.c_stats["average_nodes_visited"] = int(
            self.c_stats["total_nodes_visited"] / self.c_stats["count_searches"]
        )
        self.c_stats["average_nodes_evaluated"] = int(
            self.c_stats["total_nodes_evaluated"] / self.c_stats["count_searches"]
        )
        self.c_stats["average_nodes_cutoff"] = int(
            self.c_stats["total_nodes_cutoff"] / self.c_stats["count_searches"]
        )
        self.c_stats["average_nodes_generated"] = int(
            self.c_stats["total_nodes_generated"] / self.c_stats["count_searches"]
        )
        self.c_stats["average_depth_reached"] = (
            self.c_stats["total_depth_reached"] / self.c_stats["count_searches"]
        )
        self.c_stats["average_search_time"] = (
            self.c_stats["total_search_time"] / self.c_stats["count_searches"]
        )
        self.c_stats["percentage_searches_tim_out"] = int(
            (self.c_stats["count_tim_out"] / self.c_stats["count_searches"]) * 100
        )
        self.c_stats["nodes_per_sec"] = int(
            (self.c_stats["total_nodes_evaluated"] / self.c_stats["total_search_time"]) * 100
        )
        if self.use_quiescence:
            self.c_stats["average_q_searches"] = int(
                self.c_stats["total_q_searches"] / self.c_stats["count_searches"]
            )
        if self.use_null_moves:
            self.c_stats["average_null_cutoffs"] = (
                self.c_stats["total_null_cutoffs"] / self.c_stats["count_searches"]
            )

        print(f"Cumulative statistics for player {self.player}, {self}")
        if self.trans_table is not None:
            pretty_print_dict({**self.c_stats, **self.trans_table.get_cumulative_metrics()})
        else:
            pretty_print_dict(self.c_stats)

    def __repr__(self):
        # Try to get the name of the evaluation function, which can be a partial
        try:
            eval_name = self.evaluate.__name__
        except AttributeError:
            eval_name = self.evaluate.func.__name__
        return (
            f"AlphaBetaPlayer("
            f"player={self.player}, "
            f"max_depth={self.max_depth}, "
            f"max_time={self.max_time}, "
            f"evaluate={eval_name}, "
            f"use_null_moves={self.use_null_moves}, "
            f"use_quiescence={self.use_quiescence}, "
            f"use_transpositions={self.trans_table is not None}, "
            f"tt_size={self.trans_table.size if self.trans_table else None})"
        )

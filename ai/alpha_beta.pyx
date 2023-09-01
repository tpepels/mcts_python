# cython: language_level=3

import cython

from util import pretty_print_dict, abbreviate
from operator import itemgetter

from libc.time cimport time
from includes cimport GameState, win, loss, draw, c_uniform_random
from ai.transpos_table cimport TranspositionTable, MoveHistory
from libc.math cimport INFINITY

cdef double curr_time() except -1:
    return time(NULL)

# Search statistics to keep track of during search
cdef unsigned stat_n_eval = 0
cdef unsigned stat_killers = 0
cdef unsigned stat_q_searches = 0
cdef unsigned stat_visited = 0
cdef unsigned stat_cutoffs = 0 
cdef unsigned stat_depth_reached = 0
cdef unsigned stat_moves_gen = 0
cdef unsigned stat_tt_orders = 0
cdef unsigned count = 0
cdef bint is_interrupted = False
cdef double start_time = 0.0
cdef double start_depth_time = 0.0
cdef bint is_first = True
cdef unsigned reached
cdef unsigned root_seen = 0

# This stores a proven best_move resulting from the a/b search
cdef tuple best_move
cdef tuple anti_decisive_move

cdef dict total_search_time = {1: 0., 2: 0.}
cdef dict n_moves = {1: 0, 2: 0}
cdef dict depth_reached = {1: 0, 2: 0}
cdef dict best_value_labels = {win: "WIN", draw: "DRAW", loss: "LOSS"}


cdef double value(
    GameState state,
    double alpha,
    double beta,
    int depth,
    int max_player,
    double[:] eval_params,
    TranspositionTable trans_table,
    MoveHistory move_history,
    dict killer_moves,
    int max_d=0,
    bint root=0,
    bint use_quiescence=0,
) except -88888888: # Don't use 99999 because it's win/loss
    # Globals that keep track of relevant statistics for optimizing
    global stat_q_searches, stat_n_eval, stat_visited, stat_cutoffs, stat_moves_gen, stat_tt_orders, stat_killers
    # Globals that carry over from the calling class, mainly meant for interrupting the search.
    global count, time_limit, start_time, is_first, is_interrupted, best_move, reached, root_seen
    
    cdef bint is_max_player = state.player == max_player
    cdef double v, cur_time, new_v
    cdef list actions
    cdef tuple trans_move, killer_move, move
    cdef int m = 0
    cdef int i
    cdef int n_actions = 0
    cdef tuple depth_best_move = (0,0)
    cdef int hist_mult = 100 if is_max_player else -100
    cdef int non_losing_moves = 0
    
    reached = max(reached, max_d - depth)

    if state.is_terminal():
        v = state.get_reward(max_player)
        return v

    if not is_interrupted:
        # Do this increment after the check or the timeout error will only occur once instead of at each level
        count += 1

    # This checks if we are running out of time.
    # Don't do this for the first depth, in some games, even the lowest search depth takes a lot of time..
    if not is_interrupted and not is_first and count % 100000 == 0:  # Check every so many iterations
        if (curr_time() - start_time) > time_limit:
            is_interrupted = True

    # If we are interrupted, cut off the search and return evaluations straight away
    if depth == 0:
        stat_n_eval += 1
        v = state.evaluate(params=eval_params, player=max_player, norm=False)
        return v

    try:
        v = -INFINITY if is_max_player else INFINITY

        actions = state.evaluate_moves(state.get_legal_actions())
        n_actions = len(actions)
        # Put moves that were in the move history before the rest by giving them a really high score
        if move_history is not None:
            actions = [(actions[i][0], (actions[i][1] + (move_history.get(actions[i][0]) * hist_mult)) + c_uniform_random(0.0001, 0.001)) for i in range(n_actions)]
        else:
            actions = [(actions[i][0], actions[i][1] + c_uniform_random(0.0001, 0.001)) for i in range(n_actions)]
        
        actions.sort(key=itemgetter(1), reverse=is_max_player)
        actions = [actions[i][0] for i in range(n_actions)]
        
        stat_moves_gen += n_actions
        # If there's a killer move at this depth, and it's a legal move, then
        # we want to try this move first.
        if killer_moves is not None:
            try:
                killer_move = killer_moves.get(max_d - depth, ())
                if actions.index(killer_move) > 0:
                    actions.remove(killer_move)
                    actions.insert(0, killer_move)
                    stat_killers += 1
            except ValueError:
                pass
        
        # Check if a move already exists somewhere in the transposition table
        _, trans_move = trans_table.get(state.board_hash, max_d - depth, state.player, None)
        # If best_move exists, place it first in the list
        try:
            if actions.index(trans_move) > 0:
                actions.remove(trans_move)
                actions.insert(0, trans_move)
                stat_tt_orders += 1
        except ValueError:
            pass
        
        stat_visited += 1
        
        for m in range(n_actions):
            move = actions[m]
            # In case of a capture do a full q search
            if use_quiescence and depth == 1 and state.is_capture(move):
                new_v = quiescence(state.apply_action(move), alpha, beta, max_player, eval_params)
            else:
                new_v = value(
                    state=state.apply_action(move),
                    alpha=alpha,
                    beta=beta,
                    depth=depth - 1,
                    max_player=max_player,
                    eval_params=eval_params,
                    trans_table=trans_table,
                    move_history=move_history,
                    killer_moves=killer_moves,
                    max_d=max_d,
                    root=False,
                    use_quiescence=use_quiescence,
                )
            # Update v, alpha or beta based on the player
            if (is_max_player and new_v > v) or (not is_max_player and new_v < v):
                v = new_v
                depth_best_move = move
                # At the root remember the best move so far
                if root:
                    best_move = move
                    # If we find a winning move, just return it
                    if new_v == win:
                        return new_v

            if root:
                root_seen += 1
                # If the move is not a terminal loss, add it to the list of non-losing moves
                if new_v != loss:
                    non_losing_moves += 1

            if is_max_player:
                alpha = max(alpha, new_v)
            else:
                beta = min(beta, new_v)

            # Prune the branch
            if beta <= alpha:
                # Update the move history with a high score since we've caused a cutoff
                if move_history is not None:
                    move_history.update(move, 10)

                if killer_moves is not None:
                    killer_moves[max_d - depth] = move  # Update the killer move
                
                stat_cutoffs += 1
                break

            if is_interrupted:
                return v

        # If there's only one non-losing move, return it immediately
        if root and non_losing_moves == 1:
            # Anti-decisive move (must be the current best move)
            global anti_decisive_move
            anti_decisive_move = best_move

        return v

    finally:
        # Update the move history with the best move found in this level
        if move_history is not None:
            move_history.update(depth_best_move, 1)

        trans_table.put(
            key=state.board_hash,
            value=v,
            depth=max_d - depth,
            player=state.player,
            best_move=depth_best_move,
            board=None
        )

cdef double quiescence(GameState state, double alpha, double beta, int max_player, double[:] eval_params) except -88888888:
    # This is the quiescence function, which aims to mitigate the horizon effect by
    # conducting a more exhaustive search on volatile branches of the game tree,
    # such as those involving captures.
    global stat_q_searches
    cdef list actions
    cdef tuple move
    cdef double score
    cdef double stand_pat
    stand_pat = state.evaluate(params=eval_params, player=max_player, norm=False)
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
            stat_q_searches += 1
            # Increment the counter that tracks the number of quiescence searches performed.

            # Check if the current move is a capture. We're specifically interested in captures
            # in a quiescence search, as they tend to create major swings in the game state.

            new_state = state.apply_action(move)
            # Apply the capture move to generate the new game state that results from this move.

            score = -quiescence(new_state, -beta, -alpha, max_player, eval_params)
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

cdef class AlphaBetaPlayer:

    cdef int player
    cdef int max_depth
    cdef double[:] eval_params
    cdef bint use_quiescence
    cdef bint use_kill_moves
    cdef bint use_history
    cdef int R
    cdef double max_time
    cdef double grace_time
    cdef int interrupt_depth_limit
    cdef bint debug
    cdef list stats
    cdef list last_move_search_times
    cdef dict c_stats
    cdef str mv_str
    cdef TranspositionTable trans_table

    def __init__(
        self,
        int player,
        double max_time,
        double[:] eval_params,
        int max_depth=25,
        unsigned transposition_table_size=2**16,
        bint use_quiescence=False,
        bint use_history=True,
        bint use_kill_moves=True,
        bint use_tt=True,
        bint debug=False,
    ):
        # Identify the player
        self.player = player
        # Set the maximum depth for the iterative deepening search
        self.max_depth = max_depth
        # The evaluation function to be used
        self.eval_params = eval_params
        # Whether or not to use quiescence search
        self.use_quiescence = use_quiescence
        self.use_history = use_history
        self.use_kill_moves = use_kill_moves
        # Maximum time for a move in seconds
        self.max_time = max_time
        # The depth limit beyond which the search should always finish
        self.interrupt_depth_limit = 2
        # Whether to print debug statistics
        self.debug = debug

        # instance-level statistics
        self.stats = []
        self.c_stats = {key: 0 for key in [
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
            "total_killers",
            "total_tt_orders"
        ]}  # Initialize to 0
        self.mv_str = ', '.join(str(val) for val in self.eval_params)
        self.trans_table = None
        if use_tt:  # Store evaluation resuls
            self.trans_table = TranspositionTable(transposition_table_size)

        global total_search_time, n_moves, depth_reached
        
        total_search_time[player] = 0
        n_moves[player] = 0
        depth_reached[player] = 0
        self.last_move_search_times = []
        self.grace_time = 0

    cdef void reset_globals(self):
        # Globals that keep track of relevant statistics for optimizing
        global stat_q_searches, stat_n_eval, stat_visited, stat_cutoffs, stat_moves_gen, stat_tt_orders,stat_killers
        # Globals that carry over from the calling class, mainly meant for interrupting the search.
        global count, time_limit, start_time, is_first, is_interrupted, start_depth_time, best_move, root_seen
        global anti_decisive_move, decisive_move
        stat_q_searches = stat_n_eval = stat_visited = stat_cutoffs = stat_moves_gen = stat_killers = 0
        count = time_limit = stat_tt_orders = 0
        root_seen = start_time = start_depth_time = 0

        best_move = anti_decisive_move = None
        is_interrupted = False
        is_first = True


    cpdef tuple best_action(self, GameState state):
        cdef double v
        cdef list best_values
        cdef tuple last_best_move
        cdef double last_best_v
        cdef int stat_depth_reached
        cdef dict killer_moves = {i: None for i in range(self.max_depth + 1)} if self.use_kill_moves else None
        cdef double start_depth_time = 0
        cdef double last_search_time = 0
        cdef MoveHistory move_history = MoveHistory() if self.use_history else None

        assert state.player == self.player, "Player to move in the game is not my assigned player"
        
        self.reset_globals()

        # Globals that keep track of relevant statistics for optimizing
        global stat_q_searches, stat_n_eval, stat_visited, stat_cutoffs, stat_moves_gen, stat_tt_orders, stat_killers
        # Globals that carry over from the calling class, mainly meant for interrupting the search.
        global count, time_limit, start_time, start_depth_time, is_first, is_interrupted
        global total_search_time, n_moves, best_move, reached, root_seen

        try:
            # Keep track of search times per level
            is_first = True
            v, best_move = 0., None
            search_times = [0.0]
            best_values = []
            last_best_move = None
            last_best_v = 0
            stat_depth_reached = 0

            start_time = curr_time()
            
            for depth in range(1, self.max_depth + 1):
                if self.debug:
                    print(f"\rdepth {depth}... ", end="")
                start_depth_time = curr_time()  # Time when this depth search starts
                # How much time can be spent searching this depth
                time_limit = (self.max_time + self.grace_time) - (start_depth_time - start_time) 

                if is_interrupted:
                    break
                
                if len(self.last_move_search_times) > depth:
                    # print(f"Last search time was {self.last_move_search_times[depth]} seconds for depth {depth}.")
                    if ((start_depth_time - start_time) + (self.last_move_search_times[depth] + 1) >= (self.max_time + self.grace_time)):
                        if self.debug:
                            print(f"Because last search, it took me {self.last_move_search_times[depth]} seconds to search depth {depth}.")
                            print(f"And I have only {time_limit:.2f} seconds left to search this depth, let's leave it here for now.")
                        break
                else:
                    # Stop searching if the time limit has been exceeded or if there's not enough time to do another search
                    if ((start_depth_time - start_time) + (last_search_time * 4) >= (self.max_time + self.grace_time)):
                        if self.debug:
                            print(f"Time limit exceeded, stopping search at depth {depth}.")
                            print(f"start_depth_time - start_time = {start_depth_time - start_time:.2f} seconds, last_search_time = {last_search_time:.2f} seconds, max_time = {self.max_time:.2f} seconds, grace_time = {self.grace_time:.2f} seconds")
                        break
               
                reached = root_seen = 0

                v = value(
                    state,
                    alpha=-INFINITY,
                    beta=INFINITY,
                    depth=depth,
                    max_player=self.player,
                    max_d=depth,
                    eval_params=self.eval_params,
                    trans_table=self.trans_table,
                    move_history=move_history,
                    killer_moves=killer_moves,
                    root=True,
                    use_quiescence=self.use_quiescence,
                )

                # After the first search, we can interrupt the search
                is_first = False
                
                if not is_interrupted:
                    last_best_move = best_move
                    last_best_v = v

                    best_values.append((depth, v, best_move))
                    stat_depth_reached = depth
                    
                elif self.debug:
                        print(f"*** Search was interrupted at depth {depth} ***")
                        print(f"Time spent on this depth (reached {reached=}): {curr_time() - start_depth_time:.2f} seconds.")

                last_search_time = curr_time()
                last_search_time -= start_depth_time
                if self.debug:
                    print(f"time {(last_search_time):.3f} *** ", end="")
                # keep track of the time spent on each depth
                search_times.append(last_search_time)
                
                # Don't search further in case of an anti-decisive move
                if anti_decisive_move is not None:
                    if self.debug:
                        print(f"Anti-decisive move found: {anti_decisive_move}")
                    best_move = anti_decisive_move
                    break
                
                # Don't spend any more time on proven wins/losses
                if v in [win, loss]:
                    break

            self.grace_time += max(-self.max_time//3, self.max_time - (curr_time() - start_time))
            self.last_move_search_times = search_times.copy()

            if self.debug:
                print()
                print("." * 80)

            if is_interrupted:
                if self.debug:
                    print(f"-->> Finished best move: {last_best_move} with v={last_best_v}")
                    print(f"Interrupted best move: {best_move} with v={v}")
                    print(f"I saw {root_seen} moves at the root level (out of {len(state.get_legal_actions())}) and reached {reached} depths.")
                    print(f"So I saw {root_seen / len(state.get_legal_actions()) * 100:.2f}% of the moves at the root level.")
                    print("." * 80)
                
                best_move = last_best_move
                v = last_best_v
            
            print()
                
            if self.debug:
                total_search_time[self.player] += curr_time() - start_time
                n_moves[self.player] += 1
                depth_reached[self.player] += stat_depth_reached
                
                stat_dict = {
                    f"{self.player}_max_player": self.player,
                    f"{self.player}_eval_params": self.mv_str,
                    "nodes_best_mv_ord.": stat_tt_orders,
                    "nodes_visited": stat_visited,
                    "nodes_evaluated": stat_n_eval,
                    "T_nodes_ps": int(stat_visited / max(1, (curr_time() - start_time))),
                    "T_evals_ps": int(stat_n_eval / max(1, (curr_time() - start_time))),
                    "T_nodes_gen_ps": int(stat_moves_gen / max(1, (curr_time() - start_time))),
                    "nodes_cutoff": stat_cutoffs,
                    "nodes_avg_br_fact": int(round(stat_moves_gen / max(1, stat_visited), 0)),
                    "nodes_generated": stat_moves_gen,
                    "depth_average": int(depth_reached[self.player] / max(1, n_moves[self.player])),
                    "depth_finished": stat_depth_reached,
                    "depth": depth,
                    "search_time": (curr_time() - start_time),
                    "search_times_p.d": search_times[1:],
                    "search_time_avg": int(total_search_time[self.player] / max(1, n_moves[self.player])),
                    "search_interr.": is_interrupted,
                    "best_value": best_value_labels.get(v, v),
                    "best_move": best_move,
                    "best_values": best_values[::-1],
                    "killer_moves": stat_killers,
                    "search_grace_time": self.grace_time
                }

                if v == win:
                    stat_dict["best_value"] = "WIN"
                elif v == draw:
                    stat_dict["best_value"] = "DRAW"
                elif v == loss:
                    stat_dict["best_value"] = "LOSS"

                if self.use_quiescence:
                    stat_dict["quiescence_searches"] = stat_q_searches
                    self.c_stats["total_q_searches"] += stat_q_searches

                self.stats.append(stat_dict)

                self.c_stats["total_nodes_visited"] += stat_visited
                self.c_stats["total_nodes_evaluated"] += stat_n_eval
                self.c_stats["total_nodes_cutoff"] += stat_cutoffs
                self.c_stats["total_nodes_generated"] += stat_moves_gen
                self.c_stats["total_depth_reached"] += stat_depth_reached
                self.c_stats["total_killers"] += stat_killers
                self.c_stats["total_tt_orders"] += stat_tt_orders
                self.c_stats["max_depth_finished"] = max(stat_depth_reached, self.c_stats["max_depth_finished"])
                self.c_stats["total_search_time"] += curr_time() - start_time
                self.c_stats["count_searches"] += 1
                self.c_stats["count_tim_out"] += int(is_interrupted)
                if self.trans_table is not None:
                    stat_dict = {**stat_dict, **self.trans_table.get_metrics()}
                    self.trans_table.reset_metrics()
                pretty_print_dict(stat_dict)

            
            return best_move, v

        finally:
            self.reset_globals()

    def print_cumulative_statistics(self):
        if not self.debug or self.c_stats["count_searches"] == 0 or self.c_stats["total_search_time"] == 0:
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
            (self.c_stats["total_nodes_evaluated"] / max(self.c_stats["total_search_time"], 1)) * 100
        )
        if self.use_quiescence:
            self.c_stats["average_q_searches"] = int(
                self.c_stats["total_q_searches"] / self.c_stats["count_searches"]
            )

        print(f"Cumulative statistics for player {self.player}, {self}")
        if self.trans_table is not None:
            pretty_print_dict({**self.c_stats, **self.trans_table.get_cumulative_metrics()})
        else:
            pretty_print_dict(self.c_stats)

    def __repr__(self):
        return (
            f"a/b("
            f"p={self.player}, "
            f"max_d={self.max_depth}, "
            f"max_t={self.max_time}, "
            f"eval_params={self.mv_str}, "
            f"qs={self.use_quiescence}, "
            f"hist={self.use_history}, "
            f"kill={self.use_kill_moves})"
        )

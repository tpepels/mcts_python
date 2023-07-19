import cProfile
import pstats

from run_games import run_game, AIParams

params = {
    "n_moves_cutoff": 0,
    "m_ter": 0,
    "m_kill_s": 0.25,
    "m_imm": 0.25,
    "m_mob": 0.0,
    "m_opp_disc": 0.9,
}

p1_params = AIParams(
    ai_key="alphabeta",
    eval_key="evaluate_amazons_lieberum",
    max_player=1,
    ai_params={"max_time": 10, "debug": True},
    eval_params=params,
)
p2_params = AIParams(
    ai_key="alphabeta",
    eval_key="evaluate_amazons",
    max_player=2,
    ai_params={"max_time": 10, "debug": True},
    eval_params={},
)

run_game(game_key="amazons", game_params={"board_size": 8}, p1_params=p1_params, p2_params=p2_params)

# try:
#     # Use cProfile to run the function and save the profiling results to a file
#     cProfile.run(
#         'run_game(game_key="amazons", game_params={"board_size": 8}, p1_params=p1_params, p2_params=p2_params)',
#         "profiler_results.out",
#     )
# except KeyboardInterrupt:
#     pass

# # Create a pstats.Stats object from the output of the cProfile run
# p = pstats.Stats("profiler_results.out")

# # Sort the statistics by the cumulative time spent in the function
# p.sort_stats("cumulative")

# # Print the first 10 lines of the profiled output
# p.print_stats(20)

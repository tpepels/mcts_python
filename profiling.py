import cProfile
import pstats

from run_games import run_game, AIParams


p1_params = AIParams(
    ai_key="alphabeta",
    eval_key="evaluate_breakthrough_lorenz",
    max_player=1,
    ai_params={"max_time": 10, "debug": True, "use_kill_moves": True, "use_history": True},
    eval_params={},
)
p2_params = AIParams(
    ai_key="alphabeta",
    eval_key="evaluate_breakthrough_lorenz",
    max_player=2,
    ai_params={"max_time": 10, "debug": True, "use_history": True, "use_kill_moves": True},
    eval_params={},
)

run_game(
    game_key="breakthrough",
    game_params={},
    p1_params=p1_params,
    p2_params=p2_params,
)

# try:
#     # Use cProfile to run the function and save the profiling results to a file
#     cProfile.run(
#         'run_game(game_key="ninarow",game_params={"board_size": 9, "row_length": 5},p1_params=p1_params,p2_params=p2_params)',
#         "profiler_results.out",
#     )
# except KeyboardInterrupt:
#     pass

# # Create a pstats.Stats object from the output of the cProfile run
# p = pstats.Stats("profiler_results.out")

# # Sort the statistics by the cumulative time spent in the function
# p.sort_stats("cumulative")

# # Print the first n lines of the profiled output
# p.print_stats(30)

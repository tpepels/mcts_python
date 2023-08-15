import argparse
import cProfile
import pstats
import traceback


from run_games import AIParams, run_game

# Set up argument parser
parser = argparse.ArgumentParser(description="Run the game with or without profiling.")
parser.add_argument("--no_profile", action="store_true", help="Run without the profiler.")
args = parser.parse_args()


debug = True
algo = "mcts"
game = "breakthrough"
evaluation = "evaluate_breakthrough_lorenz"
ai_params = {"num_simulations": 500000, "debug": debug}
game_params = {}

p1_params = AIParams(
    ai_key=algo,
    eval_key=evaluation,
    max_player=1,
    ai_params=ai_params,
)
p2_params = AIParams(
    ai_key=algo,
    eval_key=evaluation,
    max_player=2,
    ai_params=ai_params,
)


def run_game_code():
    run_game(game_key=game, game_params=game_params, p1_params=p1_params, p2_params=p2_params)


if args.no_profile:
    print(" --- Running without profiler ---")
    try:
        run_game_code()
    except KeyboardInterrupt:
        pass
    except Exception as ex:
        print(str(ex))
        traceback.print_exc()
else:
    print(" --- Profiling ---")
    try:
        cProfile.run("run_game_code()", "profiler_results.out")
    except KeyboardInterrupt:
        pass
    except Exception as ex:
        print(str(ex))
        traceback.print_exc()

    p = pstats.Stats("profiler_results.out")
    p.sort_stats("tottime")
    p.print_stats(30)

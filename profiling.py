import argparse
import cProfile
import inspect
import pstats
import time
import traceback
import os
from run_games import AIParams, run_game
from util import format_time

ai_choices = ["mcts", "alphabeta"]
# Set up argument parser
parser = argparse.ArgumentParser(description="Run the game with or without profiling.")
parser.add_argument(
    "--no_profile", action="store_true", help="Run without the profiler."
)
parser.add_argument("--debug", action="store_true", help="Show debug messages.")
parser.add_argument("--pause", action="store_true", help="Pause after each turn.")
parser.add_argument(
    "--battle", action="store_true", help="Run a battle between two AIs."
)
parser.add_argument(
    "--algo",
    choices=ai_choices,
    default="alphabeta",
    help="Choose the algorithm (mcts or alphabeta).",
)
parser.add_argument(
    "--game",
    choices=[
        "amazons",
        "amazons8",
        "breakthrough",
        "ninarow59",
        "kalah66",
        "kalah86",
        "blokus",
    ],
    default="ninarow",
    help="Choose the game (amazons, breakthrough, ninarow, kalah, blokus).",
)


def get_default_params(func):
    signature = inspect.signature(func)
    param_dict = {}

    for name, param in signature.parameters.items():
        if param.default is not param.empty:
            param_dict[name] = param.default

    return param_dict


args = parser.parse_args()
# Clearing the Screen
# os.system("clear")
# c: float = 1.0,
# dyn_early_term: bool = False,
# dyn_early_term_cutoff: float = 0.9,
# early_term: bool = False,
# early_term_turns: int = 10,
# early_term_cutoff: float = 0.05,
# e_greedy: bool = False,
# roulette_epsilon: float = 0.05,
# e_g_subset: int = 20,
# imm_alpha: float = 0.4,
# imm: bool = False,
# imm_version: int = 0,
# ab_version: int = 0,
# ex_imm_D: int = 2,
# roulette: bool = False,
# epsilon: float = 0.05,
# prog_bias: bool = False,
# pb_weight: float = 0.5,
# debug: bool = False,

# Check if the game starts with "ninarow" or "kalah"
if args.game.startswith("ninarow"):
    game_name = "ninarow"
    row_length, board_size = int(args.game[-2]), int(args.game[-1])
    game_params = {"row_length": row_length, "board_size": board_size}
elif args.game.startswith("kalah"):
    game_name = "kalah"
    n_houses, init_seeds = int(args.game[-2]), int(args.game[-1])
    game_params = {"n_houses": n_houses, "init_seeds": init_seeds}
elif args.game.startswith("amazons") and args.game.endswith("8"):
    game_name = "amazons"
    board_size = int(args.game[-1])
    game_params = {
        "board_size": 8,
    }
else:
    game_name = args.game
    game_params = {}

game = game_name
if not args.battle:
    algo = args.algo
    eval_params = {}
    ai_params = {
        "max_time": 2,
        "debug": args.debug,
        # "early_term": True,
        # "early_term_turns": 10,
        # "early_term_cutoff": 0.05,
        # # "roulette": True,
        # "prog_bias": True,
        # "imm": True,
    }

    p1_params = AIParams(
        ai_key=algo,
        eval_params=eval_params,
        max_player=1,
        game_name=game_name,
        ai_params=ai_params,
    )
    p2_params = AIParams(
        ai_key=algo,
        eval_params=eval_params,
        max_player=2,
        game_name=game_name,
        ai_params=ai_params,
    )
else:
    # * Battle
    eval_params = {"a": 60, "m_lorenz": 1}

    ai_1_params = {"max_time": 15, "debug": args.debug}
    ai_2_params = {"max_time": 5, "debug": args.debug}

    p1_params = AIParams(
        ai_key="mcts",
        eval_params=eval_params,
        max_player=1,
        game_name=game_name,
        ai_params=ai_1_params,
    )
    p2_params = AIParams(
        ai_key="mcts",
        eval_params=eval_params,
        max_player=2,
        game_name=game_name,
        ai_params=ai_2_params,
    )


def run_game_code():
    run_game(
        game_key=game,
        game_params=game_params,
        p1_params=p1_params,
        p2_params=p2_params,
        pause=args.pause,
        debug=args.debug,
    )


# Time the game:
start_time = time.time()
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

end_time = time.time()

print(f" --- Game took {format_time(end_time - start_time)} seconds ---")

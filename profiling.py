import argparse
import cProfile
import pstats
import traceback
import os
from run_games import AIParams, run_game

ai_choices = ["mcts", "alphabeta"]
# Set up argument parser
parser = argparse.ArgumentParser(description="Run the game with or without profiling.")
parser.add_argument("--no_profile", action="store_true", help="Run without the profiler.")
parser.add_argument("--debug", action="store_true", help="Show debug messages.")
parser.add_argument("--pause", action="store_true", help="Pause after each turn.")
parser.add_argument("--battle", action="store_true", help="Run a battle between two AIs.")
parser.add_argument(
    "--algo",
    choices=ai_choices,
    default="alphabeta",
    help="Choose the algorithm (mcts or alphabeta).",
)
parser.add_argument(
    "--game",
    choices=["amazons", "breakthrough", "ninarow59", "kalah66", "kalah86", "blokus"],
    default="ninarow",
    help="Choose the game (amazons, breakthrough, ninarow, kalah, blokus).",
)


args = parser.parse_args()
# Clearing the Screen
os.system("clear")
# num_simulations: int = 0,
# max_time: int = 0,
# c: float = 1.0,
# dyn_early_term: bool = False,
# dyn_early_term_cutoff: float = 0.9,
# early_term: bool = False,
# early_term_turns: int = 10,
# early_term_cutoff: float = 0.05,
# e_greedy: bool = False,
# e_g_subset: int = 20,
# imm_alpha: float = 0.4,
# imm: bool = False,
# roulette: bool = False,
# epsilon: float = 0.05,
# node_priors: bool = False,
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
else:
    game_name = args.game
    game_params = {}

game = game_name
if not args.battle:
    algo = args.algo
    eval_params = {}
    ai_params = {
        "max_time": 10,
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
        ai_params=ai_params,
    )
    p2_params = AIParams(
        ai_key=algo,
        eval_params=eval_params,
        max_player=2,
        ai_params=ai_params,
    )
else:
    algo = args.algo
    eval_params = {}
    ai_1_params = {
        "num_simulations": 20_000,
        "debug": args.debug,
        # "imm": True,
        # "c": 0.6,
        # "imm_version": 0,
        # "early_term": True,
        # "early_term_turns": 10,
        # "early_term_cutoff": 0.05,
        # # "roulette": True,
        # # "roulette_epsilon": 0.05,
        # "imm_alpha": 0.6,
        # "ab_version": 4,
    }
    ai_2_params = {
        "num_simulations": 20_000,
        "debug": args.debug,
        # "imm": True,
        # "c": 0.6,
        # "imm_version": 0,
        # "early_term": True,
        # "early_term_turns": 10,
        # "early_term_cutoff": 0.05,
        # # "roulette": True,
        # # "roulette_epsilon": 0.05,
        # "imm_alpha": 0.6,
        # "ab_version": 0,
    }

    p1_params = AIParams(
        ai_key="mcts",
        eval_params=eval_params,
        max_player=1,
        ai_params=ai_1_params,
    )
    p2_params = AIParams(
        ai_key="mcts",
        eval_params=eval_params,
        max_player=2,
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

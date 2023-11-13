import os
import shutil
import traceback
from typing import Any
from run_games import AIParams, run_game_experiment
from util import redirect_print_to_log

import multiprocessing as mp

mcts_param_dicts = [
    {
        "c": 1.0,
        "dyn_early_term": False,
        "dyn_early_term_cutoff": 0.9,
        "early_term": False,
        "early_term_turns": 10,
        "early_term_cutoff": 0.05,
        "e_greedy": False,
        "roulette_epsilon": 0.05,
        "e_g_subset": 20,
        "imm_alpha": 0.4,
        "imm": False,
        "imm_version": 0,
        "ab_version": 0,
        "ex_imm_D": 2,
        "roulette": False,
        "epsilon": 0.05,
        "prog_bias": False,
        "pb_weight": 0.5,
    },
    {
        "c": 1.0,
        "dyn_early_term": True,
        "dyn_early_term_cutoff": 0.9,
        "early_term": False,
        "early_term_turns": 10,
        "early_term_cutoff": 0.05,
        "e_greedy": False,
        "roulette_epsilon": 0.05,
        "e_g_subset": 20,
        "imm_alpha": 0.4,
        "imm": False,
        "imm_version": 0,
        "ab_version": 0,
        "ex_imm_D": 2,
        "roulette": False,
        "epsilon": 0.05,
        "prog_bias": False,
        "pb_weight": 0.5,
    },
    {
        "c": 1.0,
        "dyn_early_term": False,
        "dyn_early_term_cutoff": 0.9,
        "early_term": True,
        "early_term_turns": 10,
        "early_term_cutoff": 0.05,
        "e_greedy": False,
        "roulette_epsilon": 0.05,
        "e_g_subset": 20,
        "imm_alpha": 0.4,
        "imm": False,
        "imm_version": 0,
        "ab_version": 0,
        "ex_imm_D": 2,
        "roulette": False,
        "epsilon": 0.05,
        "prog_bias": False,
        "pb_weight": 0.5,
    },
    {
        "c": 1.0,
        "dyn_early_term": False,
        "dyn_early_term_cutoff": 0.9,
        "early_term": False,
        "early_term_turns": 10,
        "early_term_cutoff": 0.05,
        "e_greedy": True,
        "roulette_epsilon": 0.05,
        "e_g_subset": 20,
        "imm_alpha": 0.4,
        "imm": False,
        "imm_version": 0,
        "ab_version": 0,
        "ex_imm_D": 2,
        "roulette": False,
        "epsilon": 0.05,
        "prog_bias": False,
        "pb_weight": 0.5,
    },
    {
        "c": 1.0,
        "dyn_early_term": False,
        "dyn_early_term_cutoff": 0.9,
        "early_term": False,
        "early_term_turns": 10,
        "early_term_cutoff": 0.05,
        "e_greedy": False,
        "roulette_epsilon": 0.05,
        "e_g_subset": 20,
        "imm_alpha": 0.4,
        "imm": True,
        "imm_version": 0,
        "ab_version": 0,
        "ex_imm_D": 2,
        "roulette": False,
        "epsilon": 0.05,
        "prog_bias": False,
        "pb_weight": 0.5,
    },
    {
        "c": 1.0,
        "dyn_early_term": False,
        "dyn_early_term_cutoff": 0.9,
        "early_term": False,
        "early_term_turns": 10,
        "early_term_cutoff": 0.05,
        "e_greedy": False,
        "roulette_epsilon": 0.05,
        "e_g_subset": 20,
        "imm_alpha": 0.4,
        "imm": False,
        "imm_version": 0,
        "ab_version": 4,
        "ex_imm_D": 2,
        "roulette": False,
        "epsilon": 0.05,
        "prog_bias": False,
        "pb_weight": 0.5,
    },
    {
        "c": 1.0,
        "dyn_early_term": False,
        "dyn_early_term_cutoff": 0.9,
        "early_term": False,
        "early_term_turns": 10,
        "early_term_cutoff": 0.05,
        "e_greedy": False,
        "roulette_epsilon": 0.05,
        "e_g_subset": 20,
        "imm_alpha": 0.4,
        "imm": False,
        "imm_version": 0,
        "ab_version": 0,
        "ex_imm_D": 2,
        "roulette": True,
        "epsilon": 0.05,
        "prog_bias": False,
        "pb_weight": 0.5,
    },
    {
        "c": 1.0,
        "dyn_early_term": False,
        "dyn_early_term_cutoff": 0.9,
        "early_term": False,
        "early_term_turns": 10,
        "early_term_cutoff": 0.05,
        "e_greedy": False,
        "roulette_epsilon": 0.05,
        "e_g_subset": 20,
        "imm_alpha": 0.4,
        "imm": False,
        "imm_version": 0,
        "ab_version": 0,
        "ex_imm_D": 2,
        "roulette": False,
        "epsilon": 0.05,
        "prog_bias": True,
        "pb_weight": 0.5,
    },
    {
        "c": 1.0,
        "dyn_early_term": True,
        "dyn_early_term_cutoff": 0.9,
        "early_term": False,
        "early_term_turns": 10,
        "early_term_cutoff": 0.05,
        "e_greedy": True,
        "roulette_epsilon": 0.05,
        "e_g_subset": 20,
        "imm_alpha": 0.4,
        "imm": True,
        "imm_version": 0,
        "ab_version": 4,
        "ex_imm_D": 2,
        "roulette": False,
        "epsilon": 0.05,
        "prog_bias": True,
        "pb_weight": 0.5,
    },
    {
        "c": 1.0,
        "dyn_early_term": False,
        "dyn_early_term_cutoff": 0.9,
        "early_term": True,
        "early_term_turns": 10,
        "early_term_cutoff": 0.05,
        "e_greedy": False,
        "roulette_epsilon": 0.05,
        "e_g_subset": 20,
        "imm_alpha": 0.4,
        "imm": True,
        "imm_version": 0,
        "ab_version": 4,
        "ex_imm_D": 2,
        "roulette": True,
        "epsilon": 0.05,
        "prog_bias": True,
        "pb_weight": 0.5,
    },
    {
        "c": 1.0,
        "dyn_early_term": False,
        "dyn_early_term_cutoff": 0.9,
        "early_term": True,
        "early_term_turns": 10,
        "early_term_cutoff": 0.05,
        "e_greedy": False,
        "roulette_epsilon": 0.05,
        "e_g_subset": 20,
        "imm_alpha": 0.4,
        "imm": True,
        "imm_version": 3,
        "ab_version": 4,
        "ex_imm_D": 2,
        "roulette": True,
        "epsilon": 0.05,
        "prog_bias": True,
        "pb_weight": 0.5,
    },
]

all_games = []
for game in ["amazons8", "amazons10", "breakthrough", "ninarow59", "kalah66", "kalah86", "blokus"]:
    # Check if the game starts with "ninarow" or "kalah"
    if game.startswith("ninarow"):
        game_name = "ninarow"
        row_length, board_size = int(game[-2]), int(game[-1])
        game_params = {"row_length": row_length, "board_size": board_size}
    elif game.startswith("kalah"):
        game_name = "kalah"
        n_houses, init_seeds = int(game[-2]), int(game[-1])
        game_params = {"n_houses": n_houses, "init_seeds": init_seeds}
    elif game.startswith("amazons"):
        game_name = "amazons"
        board_size = int(game[-1])
        if board_size == 0:
            board_size = 10
        game_params = {"board_size": board_size}
    else:
        game_name = game
        game_params = {}
    all_games.append((game_name, game_params))

mcts_fixed_params = {
    "num_simulations": 250_000,
    "debug": True,
}

ab_fixed_params = {"max_time": 10}
exp_list = []
i = 0
# Start a match for all the games to see if everything works
for game_name, game_params in all_games:
    for mcts_params in mcts_param_dicts:
        i += 1
        exp_params = (
            i,
            game_name,
            game_params,
            AIParams(
                ai_key="alphabeta",
                max_player=1,
                eval_params={},
                game_name=game_name,
                ai_params=ab_fixed_params,
            ),
            AIParams(
                ai_key="mcts",
                max_player=2,
                eval_params={},
                game_name=game_name,
                ai_params=mcts_params | mcts_fixed_params,
            ),
        )
        exp_list.append(exp_params)


def run_single_experiment(
    i: int,
    game_key: str,
    game_params: dict[str, Any],
    p1_params: AIParams,
    p2_params: AIParams,
) -> None:
    try:
        with redirect_print_to_log(f"test/{game_key}_{i}.log"):
            run_game_experiment(game_key, game_params, p1_params, p2_params)

    except Exception as e:
        with open(f"test/{game_key}_{i}.err", "a") as log_file:
            log_file.write(f"{p1_params=}\n")
            log_file.write(f"{p2_params=}\n")
            log_file.write(f"Experiment error: {e}\n")
            # Writing the traceback
            traceback.print_exc(file=log_file)

            log_file.write("========================================\n\n")
            log_file.flush()


def remake_dir(directory_path):
    shutil.rmtree(directory_path)
    os.makedirs(directory_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--n_procs", type=int, default=4, help="The number of processors to use.", required=True
    )

    args = parser.parse_args()

    remake_dir("test")

    with mp.Pool(args.n_procs) as pool:
        pool.starmap(run_single_experiment, exp_list)

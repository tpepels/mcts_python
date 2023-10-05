import argparse
from copy import deepcopy
import csv
import datetime
import glob
import itertools
import json
import multiprocessing as mp
import os
import random
import re
import time

from collections import Counter
import traceback
from typing import Any
import pandas as pd
from prettytable import PrettyTable

from run_games import AIParams, init_game, run_game_experiment
from util import redirect_print_to_log

base_path = "."


class ColName:
    N_GAMES = "n_games"
    COMPLETED_GAMES = "completed_games"
    GAME_KEY = "game_key"
    GAME_PARAMS = "game_params"
    P1_AI_KEY = "p1_ai_key"
    P1_AI_PARAMS = "p1_ai_params"
    P1_EVAL_PARAMS = "p1_eval_params"
    P2_AI_KEY = "p2_ai_key"
    P2_AI_PARAMS = "p2_ai_params"
    P2_EVAL_PARAMS = "p2_eval_params"


def read_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def expand_rows(json_file_path):
    # Read JSON data into a DataFrame
    data_dict = read_json_file(json_file_path)
    df = pd.DataFrame(data_dict)

    # Find columns that need to be expanded
    params_cols = [col for col in df.columns if col.endswith("_params") and df[col].dtype == "object"]

    res = []
    for index, row in df.iterrows():
        params_values = [
            (json.loads(row[col]) if isinstance(row[col], str) else row[col]) for col in params_cols
        ]
        params_values = [param if isinstance(param, dict) else {} for param in params_values]

        keys_values_list = [
            [(col, *item) for item in list(itertools.product([k], v if isinstance(v, list) else [v]))]
            for col, dict_ in zip(params_cols, params_values)
            for k, v in dict_.items()
        ]

        params_combinations = list(itertools.product(*keys_values_list))

        params_combinations_dicts = []
        for comb in params_combinations:
            params_dict = {}
            for col, key, value in comb:
                if col not in params_dict:
                    params_dict[col] = {}
                params_dict[col][key] = value
            params_combinations_dicts.append(params_dict)

        for params_comb in params_combinations_dicts:
            new_row = row.to_dict()
            for col in params_cols:
                if col in params_comb:
                    new_row[col] = params_comb[col]  # No json.dumps() here
                else:
                    new_row[col] = {}
            res.append(new_row)

    return res


def start_experiments_from_json(json_file_path, n_procs=4):
    """
    Read experiment configurations from a JSON file and start the experiments.

    Args:
        json_file_path (str): The path to the JSON file containing the experiment configurations.
        n_procs (int): The number of processes to be used for parallel execution. Default is 4.
    """

    # Step 1: Expand rows (if needed)
    expanded_experiment_configs = expand_rows(json_file_path)

    # Step 2: Start experiments using multiprocessing
    for exp_dict in expanded_experiment_configs:
        with mp.Pool(processes=n_procs) as pool:
            async_result, sheet_name = run_new_experiment(exp_dict, pool)

            while not async_result.ready():
                # Update running experiment status every 10 seconds
                update_running_experiment_status(exp_name=sheet_name)
                time.sleep(10)  # Sleep for 10 seconds

            # Now the experiment is done, update the status one last time
            update_running_experiment_status(exp_name=sheet_name)


def run_new_experiment(exp_dict, pool):
    game_params = exp_dict[ColName.GAME_PARAMS]
    game_name = exp_dict[ColName.GAME_KEY]
    game = init_game(game_name, game_params=game_params)

    p1_params = AIParams(
        ai_key=exp_dict[ColName.P1_AI_KEY],
        ai_params=exp_dict[ColName.P1_AI_PARAMS],
        max_player=1,
        eval_params=exp_dict[ColName.P1_EVAL_PARAMS],
        transposition_table_size=game.transposition_table_size,
    )
    p2_params = AIParams(
        ai_key=exp_dict[ColName.P2_AI_KEY],
        ai_params=exp_dict[ColName.P2_AI_PARAMS],
        max_player=2,
        eval_params=exp_dict[ColName.P2_EVAL_PARAMS],
        transposition_table_size=game.transposition_table_size,
    )

    exp_name = f"{game}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    start_game = 0

    del game
    n_games = exp_dict[ColName.N_GAMES]
    print(f"starting experiment {exp_name}")

    games_params = []
    for i in range(n_games):
        if i < n_games / 2:
            games_params.append(
                (game_name, game_params, deepcopy(p1_params), deepcopy(p2_params), False, exp_name)
            )
        else:
            new_p1_params = deepcopy(p2_params)
            new_p2_params = deepcopy(p1_params)
            new_p1_params.max_player = 1
            new_p2_params.max_player = 2
            games_params.append((game_name, game_params, new_p1_params, new_p2_params, True, exp_name))

    random.shuffle(games_params)
    games_params = [(i, *params) for i, params in enumerate(games_params)]
    games_params = [game for game in games_params if game[0] >= start_game]

    async_result = pool.starmap_async(run_single_experiment, games_params)

    time.sleep(1)

    return async_result, exp_name


def update_running_experiment_status(exp_name):
    completed_games = 0
    ai_stats = Counter()  # To hold cumulative statistics per AI
    path_to_log = f"{base_path}/log/games/{exp_name}"
    os.makedirs(f"{path_to_log}", exist_ok=True)
    log_files = glob.glob(f"{path_to_log}/?.log")

    # Open CSV file in append mode
    with open(f"{path_to_log}/_results.csv", "w", newline="") as f:
        writer = csv.writer(f)

        for log_file in log_files:
            with open(log_file, "r") as log_f:
                log_contents = log_f.readlines()

                if len(log_contents) < 3:
                    continue

                game_number = log_file.split("/")[-1].split(".")[0]  # Get game number from log file name
                if (
                    "Experiment completed" in log_contents[-1]
                ):  # Assuming "Experiment completed" is second to last line
                    completed_games += 1
                    # Assuming the last line is "Game Over. Winner: Px [px_params]"
                    winner_info = log_contents[-2]  # Last line
                    winner_params = re.search(r"\[(.*)\]", winner_info).group(
                        1
                    )  # Extracts string between [] and ]
                    if "Game Over. Winner: P1" in winner_info or "Game Over. Winner: P2" in winner_info:
                        writer.writerow([exp_name, game_number, winner_params])
                        ai_stats[winner_params] += 1  # Update AI statistics
                    else:
                        writer.writerow([exp_name, game_number, "Draw"])
                elif "Experiment error" in log_contents[-1]:  # Assuming "Experiment error" is the last line
                    writer.writerow([exp_name, game_number, "Error"])

    # Print cumulative statistics per AI to the screen
    print_stats = PrettyTable(["AI", f"Win % (Games: {completed_games})"])
    for ai, wins in ai_stats.items():
        print_stats.add_row([ai, f"{(wins / completed_games) * 100: .2f}"])

    print(f"{exp_name} - {completed_games} games completed.")
    print("***-" * 20)
    print(print_stats)


def run_single_experiment(
    i: int,
    game_key: str,
    game_params: dict[str, Any],
    p1_params: AIParams,
    p2_params: AIParams,
    players_switched: bool,
    exp_name: str,
) -> None:
    """Run a single game experiment and log the results in the worksheet.

    Args:
        i (int): The game number.
        game_key (str): The key for the game.
        game_params (dict[str, Any]): The parameters for the game.
        p1_params (AIParams): The parameters for player 1's AI.
        p2_params (AIParams): The parameters for player 2's AI.
        worksheet_name (str): The name of the sheet to place the results in.
        players_switched (bool): Whether player 1 and 2 are switched.
        worksheet_name (str): String used to identify the experiment
    """

    try:
        # TODO Keep track of all statistics of the game
        with redirect_print_to_log(f"{base_path}/log/games/{exp_name}/{i}.log"):
            _, _, _, _, result = run_game_experiment(game_key, game_params, p1_params, p2_params)

        with open(f"{base_path}/log/games/{exp_name}/{i}.log", "a") as log_file:
            # Write a status message to the log file
            log_file.write("Experiment completed")

    except Exception as e:
        with open(f"{base_path}/log/games/{exp_name}/{i}.log", "a") as log_file:
            # Writing the traceback
            traceback.print_exc(file=log_file)
            log_file.write(f"Experiment error: {e}")
            log_file.flush()

    # Keep track of results per AI not for p1/p2
    # Map game result to player's outcomes
    results_map = {
        1: (1, 0),
        2: (0, 1),
        0: (0, 0),
    }
    p1_result, p2_result = results_map[result]
    # If players were switched, swap the results
    if players_switched:
        p1_result, p2_result = p2_result, p1_result


def main():
    parser = argparse.ArgumentParser(description="Start experiments based on JSON config.")
    parser.add_argument("--n_procs", type=int, default=4, help="Number of processes for parallel execution.")
    parser.add_argument("--base_path", type=str, default=".", help="Base directory to create log files.")
    parser.add_argument(
        "--json_file", type=str, required=True, help="JSON file containing experiment configurations."
    )

    args = parser.parse_args()
    global base_path
    base_path = args.base_path

    # Validate and create the base directory for logs if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Read the JSON file to see if it exists and is a valid JSON
    try:
        with open(args.json_file, "r") as f:
            json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    # Start experiments
    start_experiments_from_json(json_file_path=args.json_file, n_procs=args.n_procs)


if __name__ == "__main__":
    main()

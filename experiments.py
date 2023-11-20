import argparse
from copy import deepcopy
import csv
import datetime
import glob
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import time

from collections import Counter
import traceback
from typing import Any
import pandas as pd
from prettytable import PrettyTable

from run_games import AIParams, init_game, run_game_experiment
from util import redirect_print_to_log


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
    RANDOM_OPENINGS = "random_openings"


def read_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def expand_rows(json_file_path):
    """
    Expands rows in a DataFrame based on "_params" columns containing JSON objects or dictionaries.

    This function reads a JSON file into a Pandas DataFrame. It then identifies the columns that
    have names ending with "_params" and contain JSON objects or dictionaries as their values.
    For each row in the DataFrame, it generates combinations of parameters from these identified columns
    and creates new rows accordingly.

    Parameters:
        json_file_path (str): The path of the JSON file to read.

    Returns:
        list: A list of dictionaries where each dictionary represents a new row in the DataFrame.

    Examples:

        1. JSON file content:
        [
            {
                "id": 1,
                "name": "Alice",
                "settings_params": "{\"color\": [\"red\", \"green\"], \"size\": \"small\"}"
            },
            {
                "id": 2,
                "name": "Bob",
                "settings_params": "{\"color\": \"blue\"}"
            }
        ]

        expand_rows("path/to/json/file.json")
        Output:
        [
            {'id': 1, 'name': 'Alice', 'settings_params': {'color': 'red', 'size': 'small'}},
            {'id': 1, 'name': 'Alice', 'settings_params': {'color': 'green', 'size': 'small'}},
            {'id': 2, 'name': 'Bob', 'settings_params': {'color': 'blue'}}
        ]

        2. JSON file content:
        [
            {
                "id": 1,
                "settings_params": "{\"flag\": true}"
            }
        ]

        expand_rows("path/to/another/json/file.json")
        Output:
        [
            {'id': 1, 'settings_params': {'flag': true}}
        ]
    """
    # Read JSON data into a DataFrame
    data_dict = read_json_file(json_file_path)
    df = pd.DataFrame(data_dict)

    # Find columns that need to be expanded
    params_cols = [
        col
        for col in df.columns
        if col.endswith("_params") and df[col].dtype == "object"
    ]

    res = []
    for _, row in df.iterrows():
        params_values = [
            (json.loads(row[col]) if isinstance(row[col], str) else row[col])
            for col in params_cols
        ]
        params_values = [
            param if isinstance(param, dict) else {} for param in params_values
        ]

        keys_values_list = [
            [
                (col, *item)
                for item in list(
                    itertools.product([k], v if isinstance(v, list) else [v])
                )
            ]
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


def start_experiments_from_json(
    json_file_path, n_procs=4, count_only=False, agg_loc=None
):
    """
    Read experiment configurations from a JSON file and start the experiments.

    Args:
        json_file_path (str): The path to the JSON file containing the experiment configurations.
        n_procs (int): The number of processes to be used for parallel execution. Default is 4.
    """

    # Step 1: Expand rows (if needed)
    expanded_experiment_configs = expand_rows(json_file_path)
    print(f"Starting {len(expanded_experiment_configs)} experiments.")
    if count_only:
        return
    # Step 2: Start experiments using multiprocessing
    for exp_dict in expanded_experiment_configs:
        with mp.Pool(processes=n_procs) as pool:
            async_result, exp_name = run_new_experiment(exp_dict, pool)

            while not async_result.ready():
                time.sleep(300)
                # Update running experiment status every 10 seconds
                update_running_experiment_status(exp_name=exp_name)

            tables[exp_name]["end_time"] = datetime.datetime.now()

            time.sleep(10)
            # Now the experiment is done, update the status one last time
            update_running_experiment_status(exp_name=exp_name)
            # Aggregate results
            if agg_loc is not None:
                aggregate_csv_results(agg_loc)


def run_new_experiment(exp_dict, pool):
    game_params = exp_dict[ColName.GAME_PARAMS]
    game_name = exp_dict[ColName.GAME_KEY]
    random_openings = int(exp_dict.get(ColName.RANDOM_OPENINGS, 0))
    game = init_game(game_name, game_params=game_params)

    p1_params = AIParams(
        ai_key=exp_dict[ColName.P1_AI_KEY],
        ai_params=exp_dict[ColName.P1_AI_PARAMS],
        max_player=1,
        eval_params=exp_dict[ColName.P1_EVAL_PARAMS],
        game_name=game_name,
        transposition_table_size=game.transposition_table_size,
    )
    p2_params = AIParams(
        ai_key=exp_dict[ColName.P2_AI_KEY],
        ai_params=exp_dict[ColName.P2_AI_PARAMS],
        max_player=2,
        eval_params=exp_dict[ColName.P2_EVAL_PARAMS],
        game_name=game_name,
        transposition_table_size=game.transposition_table_size,
    )

    exp_name = f"{game}_{exp_dict[ColName.P1_AI_KEY]}_{exp_dict[ColName.P2_AI_KEY]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    start_game = 0

    del game
    n_games = exp_dict[ColName.N_GAMES]
    print(f"starting experiment {exp_name}")

    games_params = []
    for i in range(n_games):
        if i < n_games / 2:
            games_params.append(
                (
                    game_name,
                    game_params,
                    deepcopy(p1_params),
                    deepcopy(p2_params),
                    exp_name,
                    base_path,
                    random_openings,
                )
            )
        else:
            new_p1_params = deepcopy(p2_params)
            new_p2_params = deepcopy(p1_params)
            new_p1_params.max_player = 1
            new_p2_params.max_player = 2
            games_params.append(
                (
                    game_name,
                    game_params,
                    new_p1_params,
                    new_p2_params,
                    exp_name,
                    base_path,
                    random_openings,
                )
            )

    random.shuffle(games_params)
    games_params = [(i, *params) for i, params in enumerate(games_params)]
    games_params = [game for game in games_params if game[0] >= start_game]

    async_result = pool.starmap_async(run_single_experiment, games_params)
    """
    Write the experiment configuration as a header to a CSV file in the log directory.
    So we can easily find results of a specific experiment.
    """
    path_to_result = os.path.join(base_path, "results", "experiments", exp_name)
    path_to_log = os.path.join(base_path, "log", "games", exp_name)
    os.makedirs(path_to_log, exist_ok=True)
    os.makedirs(path_to_result, exist_ok=True)

    tables[exp_name] = {}
    description_str = f"{exp_name=}  --  {game_name=}\n"
    description_str += f"{game_params=}\n"
    description_str += f"{p1_params=}\n"
    description_str += f"{p2_params=}\n"
    description_str += f"{n_games=}\n"
    description_str += f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n     -------- \n"
    tables[exp_name]["description"] = description_str
    tables[exp_name]["start_time"] = datetime.datetime.now()
    time.sleep(1)

    return async_result, exp_name


def update_running_experiment_status(exp_name, print_tables=True):
    completed_games = 0
    error_games = 0
    draws = 0
    ai_stats = Counter()  # To hold cumulative statistics per AI

    path_to_result = os.path.join(base_path, "results", "experiments", exp_name)
    path_to_log = os.path.join(base_path, "log", "games", exp_name)
    os.makedirs(path_to_log, exist_ok=True)
    os.makedirs(path_to_result, exist_ok=True)

    log_files = glob.glob(f"{path_to_log}/*.log")

    # Open CSV file in write mode (it needs to be overwritten every time)
    with open(os.path.join(path_to_result, "_results.csv"), "w", newline="") as f:
        if exp_name in tables:
            f.write(tables[exp_name]["description"])

        writer = csv.writer(f)

        for log_file in log_files:
            with open(log_file, "r") as log_f:
                log_contents = log_f.readlines()

                if len(log_contents) < 3:
                    continue

                game_number = log_file.split("/")[-1].split(".")[
                    0
                ]  # Get game number from log file name
                if (
                    "Experiment completed" in log_contents[-1]
                ):  # Assuming "Experiment completed" is second to last line
                    completed_games += 1
                    # Assuming the last line is "Game Over. Winner: Px [px_params]"
                    winner_info = log_contents[-2]
                    winner_params = re.search(r"\[(.*)\]", winner_info).group(1)

                    if (
                        "Game Over. Winner: P1" in winner_info
                        or "Game Over. Winner: P2" in winner_info
                    ):
                        loser_info = log_contents[-3]
                        loser_params = re.search(r"\[(.*)\]", loser_info).group(1)

                        # Make sure there's a line for each player
                        if winner_params not in ai_stats:
                            ai_stats[winner_params] = 0
                        if loser_params not in ai_stats:
                            ai_stats[loser_params] = 0

                        writer.writerow([game_number, winner_params])
                        ai_stats[winner_params] += 1  # Update AI statistics
                    else:
                        draws += 1
                        writer.writerow([game_number, "Draw"])
                elif (
                    "Experiment error" in log_contents[-1]
                ):  # Assuming "Experiment error" is the last line
                    writer.writerow([game_number, "Error"])
                    error_games += 1

    # Write cumulative table to separate CSV file
    with open(f"{path_to_result}/_cumulative_stats.csv", "w", newline="") as f:
        if exp_name in tables:
            f.write(tables[exp_name]["description"])
        writer = csv.writer(f)
        writer.writerow(["AI", "Win %", "95% C.I.", "# Games"])

        Z = 1.96
        for ai, wins in ai_stats.items():
            if completed_games > 0:
                win_rate = wins / completed_games
                ci_width = Z * math.sqrt((win_rate * (1 - win_rate)) / completed_games)
                lower_bound = (win_rate - ci_width) * 100
                upper_bound = (win_rate + ci_width) * 100
                writer.writerow(
                    [
                        ai,
                        f"{win_rate * 100:.2f}",
                        f"±{upper_bound - lower_bound:.2f}",
                        completed_games,
                    ]
                )
            else:
                writer.writerow([ai, "N/A", "N/A", 0])

    if print_tables:
        # Print cumulative statistics per AI to the screen
        print_stats = PrettyTable(
            [
                f"AI ({exp_name})",
                f"Win % (Games: {completed_games}, Errors: {error_games}, Draws: {draws})",
                "95% C.I.",
            ]
        )
        # Z-score for 95% confidence interval
        Z = 1.96
        # Add rows to the table
        for ai, wins in ai_stats.items():
            win_rate = wins / completed_games
            ci_width = Z * math.sqrt((win_rate * (1 - win_rate)) / completed_games)
            lower_bound = (win_rate - ci_width) * 100
            upper_bound = (win_rate + ci_width) * 100
            print_stats.add_row(
                [ai, f"{win_rate * 100:.2f}", f"±{upper_bound - lower_bound:.2f}"]
            )

        # Keep track of all experiments, also the finished ones to print
        tables[exp_name]["table"] = print_stats
        print("\n\n\n\n")
        print("-" * 20, end="")
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), end="")
        print("-" * 20)
        print("\n")
        for _, v in tables.items():
            print(v["description"])
            print(v["table"])
            if "end_time" in v:
                print(f"Finished: {v['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Duration: {v['end_time'] - v['start_time']}")
                if completed_games > 0:
                    print(
                        f"Average time per game: {(v['end_time'] - v['start_time']) / float(completed_games)}"
                    )


tables = {}


def generate_ascii_art(text):
    # Define the banner style for each character
    # For simplicity, let's use a basic style. You can expand this dictionary for a more detailed representation.
    banner = {
        "A": ["   A     ", "  A A    ", " A   A   ", "AAAAAAA  ", "A     A  "],
        "B": ["BBBB     ", "B    B   ", "BBBBB    ", "B    B   ", "BBBBB    "],
        "C": [" CCCCC   ", "C        ", "C        ", "C        ", " CCCCC   "],
        "E": ["EEEEE    ", "E        ", "EEEEE    ", "E        ", "EEEEE    "],
        "F": ["FFFFF    ", "F        ", "FFFF     ", "F        ", "F        "],
        "I": ["IIIII    ", "  I      ", "  I      ", "  I      ", "IIIII    "],
        "M": ["M     M  ", "MM   MM  ", "M M M M  ", "M  M  M  ", "M     M  "],
        "N": ["N     N  ", "NN    N  ", "N N   N  ", "N  N  N  ", "N     N  "],
        "P": ["PPPP     ", "P    P   ", "PPPP     ", "P        ", "P        "],
        "R": ["RRRR     ", "R    R   ", "RRRR     ", "R R      ", "R   R    "],
        "S": [" SSSSS   ", "S        ", " SSSS    ", "      S  ", " SSSSS   "],
        "T": ["TTTTTT   ", "  T      ", "  T      ", "  T      ", "  T      "],
        "X": ["X   X    ", " X X     ", "  X      ", " X X     ", "X   X    "],
        " ": ["         ", "         ", "         ", "         ", "         "],
        "!": ["  !      ", "  !      ", "  !      ", "         ", "  !      "],
    }

    # Generate the ASCII art line by line
    for i in range(5):
        line = ""
        for char in text:
            if char.upper() in banner:
                line += banner[char.upper()][i]
        print(line)


def aggregate_csv_results(output_file):
    files = []
    experiments_path = os.path.join(base_path, "results", "experiments")
    for dir_name in os.listdir(experiments_path):
        full_dir_path = os.path.join(experiments_path, dir_name)
        if os.path.isdir(full_dir_path):
            files.append(os.path.join(full_dir_path, "_results.csv"))

    with open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(
            [
                "Experiment Name",
                "Date-Time",
                "Game Name",
                "Game Parameters",
                "p1_params",
                "p2_params",
                "AI1",
                "Param1",
                "Win_rate",
                "AI2",
                "Param2",
                "Win_rate",
                "± 95% C.I.",
                "No. Games",
            ]
        )
        aggregated_rows = []

        for file in files:
            ai_stats = {}
            total_games = 0
            metadata = {}

            with open(file, "r") as infile:
                lines = infile.readlines()

                # Check for enough lines in the file
                if len(lines) < 8:
                    print(f"Skipping {file} due to insufficient lines.")
                    continue

                # Parse metadata using regular expressions
                try:
                    metadata["exp_name"] = re.search(
                        r"exp_name='(.*?)'", lines[0]
                    ).group(1)
                    metadata["date_time"] = re.search(
                        r"(\d{8}_\d{6})", metadata["exp_name"]
                    ).group(1)
                    metadata["game_name"] = re.search(
                        r"game_name='(.*?)'", lines[0]
                    ).group(1)
                    metadata["game_params"] = re.search(
                        r"game_params\s*=\s*(\{.*?\})", lines[1]
                    ).group(1)
                    metadata["p1_params"] = lines[2].split("=", 1)[1].strip()
                    metadata["p2_params"] = lines[3].split("=", 1)[1].strip()
                except (AttributeError, IndexError) as e:
                    print(f"Error parsing metadata for {file}.")
                    print("Lines causing issues:")
                    print(lines[:4])  # Print the lines causing the issue
                    continue

                for line in lines[7:]:
                    _, ai_config = line.strip().split(",", 1)
                    ai_config_cleaned = sort_parameters(ai_config.strip().strip('"'))
                    if ai_config_cleaned == "Draw":
                        continue
                    total_games += 1
                    ai_stats[ai_config_cleaned] = ai_stats.get(ai_config_cleaned, 0) + 1

            Z = 1.96
            ai_results = []

            for ai_config, wins in ai_stats.items():
                win_rate = wins / total_games
                ci_width = Z * math.sqrt((win_rate * (1 - win_rate)) / total_games)
                lower_bound = (win_rate - ci_width) * 100
                upper_bound = (win_rate + ci_width) * 100
                ai_results.append(
                    (
                        ai_config,
                        f"{win_rate * 100:.2f}",
                        f"±{upper_bound - lower_bound:.2f}",
                    )
                )

            # Sort ai_results based on ai_config to ensure consistent order
            ai_results.sort(key=lambda x: (len(x[0]), x[0]))

            # Construct the row for this file
            row = [
                metadata["exp_name"],
                metadata["date_time"],
                metadata["game_name"],
                metadata["game_params"],
                metadata["p1_params"],
                metadata["p2_params"],
            ]

            ai1_diffs, ai2_diffs = {}, {}
            if len(ai_results) == 2:
                ai1_diffs, ai2_diffs = extract_ai_param_diffs(
                    ai_results[0][0], ai_results[1][0]
                )
                row.extend(
                    [
                        ai_results[0][0],
                        dict_to_str(ai1_diffs),
                        ai_results[0][1],
                        ai_results[1][0],
                        dict_to_str(ai2_diffs),
                        ai_results[1][1],
                        ai_results[0][2],
                        total_games,
                    ]
                )
            elif len(ai_results) == 1:
                row.extend(
                    [
                        ai_results[0][0],
                        "N/A",
                        ai_results[0][1],
                        "N/A",
                        "N/A",
                        "N/A",
                        ai_results[0][2],
                    ]
                )
            else:
                row.extend(["N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])

            aggregated_rows.append(row)

        # Sort the aggregated rows by the AI1 win rate
        aggregated_rows.sort(key=lambda row: float(row[8]), reverse=True)
        # Write the sorted rows to the output file
        for row in aggregated_rows:
            writer.writerow(row)


def sort_parameters(ai_config):
    params = ai_config.split(", ")
    params.sort()
    return ", ".join(params)


def extract_ai_param_diffs(ai1, ai2):
    dict1 = {m.group(1): m.group(2) for m in re.finditer(r"(\w+):([\d.]+)", ai1)}
    dict2 = {m.group(1): m.group(2) for m in re.finditer(r"(\w+):([\d.]+)", ai2)}

    diffs1 = {}
    diffs2 = {}

    for k in set(dict1.keys()).union(dict2.keys()):  # Union of keys from both dicts
        v1 = dict1.get(k)
        v2 = dict2.get(k)

        if v1 != v2:
            diffs1[k] = v1 or "N/A"  # Assign 'N/A' if the value is None
            diffs2[k] = v2 or "N/A"

    return diffs1, diffs2


def dict_to_str(d):
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def run_single_experiment(
    i: int,
    game_key: str,
    game_params: dict[str, Any],
    p1_params: AIParams,
    p2_params: AIParams,
    exp_name: str,
    base_path: str = ".",
    random_openings: int = 0,
) -> None:
    """Run a single game experiment and log the results in the worksheet.

    Args:
        i (int): The game number.
        game_key (str): The key for the game.
        game_params (dict[str, Any]): The parameters for the game.
        p1_params (AIParams): The parameters for player 1's AI.
        p2_params (AIParams): The parameters for player 2's AI.
        worksheet_name (str): The name of the sheet to place the results in.
        worksheet_name (str): String used to identify the experiment
    """

    log_path = os.path.join(base_path, "log", "games", exp_name, f"{i}.log")
    try:
        with redirect_print_to_log(log_path):
            run_game_experiment(
                game_key, game_params, p1_params, p2_params, random_openings
            )

        with open(log_path, "a") as log_file:
            # Write a status message to the log file
            log_file.write("Experiment completed")

    except Exception as e:
        with open(log_path, "a") as log_file:
            # Writing the traceback
            traceback.print_exc(file=log_file)
            log_file.write(f"Experiment error: {e}")
            log_file.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Start experiments based on JSON config."
    )
    parser.add_argument(
        "-n",
        "--n_procs",
        type=int,
        default=4,
        help="Number of processes for parallel execution.",
    )
    parser.add_argument(
        "-b",
        "--base_path",
        type=str,
        default=".",
        help="Base directory to create log files.",
    )
    parser.add_argument(
        "-j",
        "--json_file",
        type=str,
        help="JSON file containing experiment configurations.",
    )
    parser.add_argument(
        "-a",
        "--aggregate_results",
        help="Aggregate results of a previous experiment to an aggregate file. If not provided, the name of the JSON file will be used with a .csv extension and saved in the home directory.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-c",
        "--clean",
        help="Clean the log directory before starting experiments.",
        action="store_true",
    )
    parser.add_argument(
        "--count_only",
        help="Count the total number of experiments that will be run",
        action="store_true",
    )
    args = parser.parse_args()

    if not (args.json_file or args.aggregate_results):
        parser.error(
            "Either --json_file should be set OR --aggregate_resultsshould be enabled."
        )

    global base_path

    if args.aggregate_results and not args.json_file:
        base_path = args.base_path
        # If no json file was given, just aggregate the results
        print(f"Aggregating results from {base_path} to {args.aggregate_results}")
        aggregate_csv_results(args.aggregate_results)
        return

    # Include the experiment file in the base_path
    base_path = os.path.join(
        args.base_path, os.path.splitext(os.path.basename(args.json_file))[0]
    )
    print("Base path:", base_path)
    # Use the name of the JSON file with a .csv extension if --aggregate_results is not provided.
    agg_loc = os.path.join(
        base_path, os.path.splitext(os.path.basename(args.json_file))[0] + ".csv"
    )
    # Validate and create the base directory for logs if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    elif args.clean:
        # Check if agg_loc file exists and rename it by appending a timestamp
        if os.path.exists(agg_loc):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = os.path.join(
                base_path,
                os.path.splitext(os.path.basename(args.json_file))[0]
                + f"_{timestamp}.csv",
            )
            os.rename(agg_loc, new_name)
            print(f"Renamed {agg_loc} to {new_name}")

        # Remove the log directory from the base_path
        log_path = os.path.join(base_path, "log")
        if os.path.exists(log_path):
            print(f"Removing {log_path}")
            shutil.rmtree(log_path)
        else:
            print(f"{log_path} does not exist.")

        result_path = os.path.join(base_path, "results")
        if os.path.exists(result_path):
            print(f"Removing {result_path}")
            shutil.rmtree(result_path)
        else:
            print(f"{result_path} does not exist.")

    # Read the JSON file to see if it exists and is a valid JSON
    try:
        with open(args.json_file, "r") as f:
            json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    # Start experiments
    start_experiments_from_json(
        json_file_path=args.json_file,
        n_procs=args.n_procs,
        count_only=args.count_only,
        agg_loc=agg_loc,
    )
    if args.count_only:
        return
    print(f"Aggregating results from {base_path} to {agg_loc}")
    aggregate_csv_results(agg_loc)

    text = "Experiment Finished!"
    generate_ascii_art(text)


if __name__ == "__main__":
    main()

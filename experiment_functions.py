from collections import Counter
import csv
import datetime
import glob
import itertools
import json
import math
import traceback
import os
import re
import pandas as pd
from prettytable import PrettyTable

from util import format_time


class ColName:
    # Column names for csv files
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


def aggregate_csv_results(output_file, base_path):
    files = []
    experiments_path = os.path.join(base_path, "results")
    stuck_games_list = []
    for dir_name in os.listdir(experiments_path):
        # Extract the game name from the directory name
        game_name = dir_name.split("_")[0]
        full_dir_path = os.path.join(experiments_path, dir_name)

        if os.path.isdir(full_dir_path):
            files.append(os.path.join(full_dir_path, f"{game_name}_results.csv"))

    print(f"Aggregating results from the {len(files)} results files.")

    try:
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
                print(f"processing {file}")
                ai_stats = {}
                total_games = 0
                metadata = {}
                try:
                    with open(file, "r") as infile:
                        lines = infile.readlines()

                        # Check for enough lines in the file
                        if len(lines) < 8:
                            print(f"Skipping {file} due to insufficient lines.")
                            # If a file is empty and has not received data for the last 10 minutes it is stuck
                            if (
                                os.path.getsize(file) == 0
                                and (
                                    datetime.datetime.now() - datetime.datetime.fromtimestamp(os.path.getmtime(file))
                                ).seconds
                                > 600
                            ):
                                print(f"File {file} is empty and has not received data for the last 10 minutes.")
                                stuck_games_list.append(file)
                            continue

                        # Parse metadata using regular expressions
                        try:
                            metadata["exp_name"] = re.search(r"exp_name='(.*?)'", lines[0]).group(1)
                            metadata["date_time"] = re.search(r"(\d{4}_\d{4})", metadata["exp_name"]).group(1)
                            metadata["game_name"] = re.search(r"game_name='(.*?)'", lines[0]).group(1)
                            metadata["game_params"] = re.search(r"game_params\s*=\s*(\{.*?\})", lines[1]).group(1)
                            metadata["p1_params"] = lines[2].split("=", 1)[1].strip()
                            metadata["p2_params"] = lines[3].split("=", 1)[1].strip()
                        except (AttributeError, IndexError) as e:
                            print(f"Error parsing metadata for {file}.")
                            print("Lines causing issues:")
                            print(lines[:4])  # Print the lines causing the issue
                            print(f"Error: {e}")
                            traceback.print_exc()
                            continue

                        for line in lines[7:]:
                            _, ai_config = line.strip().split(",", 1)
                            ai_config_cleaned = sort_parameters(ai_config.strip().strip('"'))
                            if (
                                ai_config_cleaned == "Draw"
                                or ai_config_cleaned == "Stuck"
                                or ai_config_cleaned == "Error"
                            ):
                                continue
                            total_games += 1
                            ai_stats[ai_config_cleaned] = ai_stats.get(ai_config_cleaned, 0) + 1

                        print(f"  The total number of games is {total_games}")
                except Exception as e:
                    # Because multiple processes are writing to the same file, it's possible that the file is not available to read
                    print(f"Error reading {file}: {e}")
                    continue

                Z = 1.96
                ai_results = []

                for ai_config, wins in ai_stats.items():
                    win_rate = wins / total_games
                    ci_width = Z * math.sqrt((win_rate * (1 - win_rate)) / total_games)
                    ai_results.append(
                        (
                            ai_config,
                            f"{win_rate * 100:.2f}",
                            f"±{ci_width * 100:.2f}",
                        )
                    )
                    print(f"  {ai_config}: {win_rate * 100:.2f}% ±{ci_width * 100:.2f}%")

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
                    print(f"There are two AI results for {file}.")
                    ai1_diffs, ai2_diffs = extract_ai_param_diffs(ai_results[0][0], ai_results[1][0])
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
                    print(f"Only one AI result for {file}.")
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
                    print(f"Error: {file} has {len(ai_results)} results. Skipping.")
                    row.extend(["N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])

                aggregated_rows.append(row)

            if len(aggregated_rows) > 0:
                print(f"Writing {len(aggregated_rows)} rows to {output_file}.")

                try:
                    aggregated_rows.sort(key=lambda row: float(row[8]), reverse=False)
                except ValueError as e:
                    print(f"Error sorting rows: {e}, {file}")

                # Write the sorted rows to the output file
                for row in aggregated_rows:
                    writer.writerow(row)
    except Exception as e:
        print(f"Error aggregating results: {e}, skipping {file}")
        # Print the stack trace so we can find the error
        traceback.print_exc()

    return stuck_games_list


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


def read_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


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


def expand_rows(json_file_path):
    # Read JSON data into a DataFrame
    data_dict = read_json_file(json_file_path)
    df = pd.DataFrame(data_dict)

    # Find columns that need to be expanded
    params_cols = [col for col in df.columns if col.endswith("_params") and df[col].dtype == "object"]

    res = []
    for _, row in df.iterrows():
        params_values = [(json.loads(row[col]) if isinstance(row[col], str) else row[col]) for col in params_cols]
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


import os


def prepare_paths(base_path, exp_name):
    """
    Ensure that the necessary directories for logging and results exist for a given experiment.

    Args:
        base_path (str): The base directory where experiment logs and results are stored.
        exp_name (str): The name of the experiment, used to create specific subdirectories.

    Returns:
        tuple: A tuple containing the paths to the result and log directories for the experiment.
    """
    # Construct paths for the experiment's results and log directories
    path_to_result = os.path.join(base_path, "results", exp_name)
    path_to_log = os.path.join(base_path, "log", exp_name)

    # Ensure the directories exist, creating them if necessary
    os.makedirs(path_to_log, exist_ok=True)
    os.makedirs(path_to_result, exist_ok=True)

    # Return the paths for further use
    return path_to_result, path_to_log


def update_running_experiment_status(tables, base_path, total_games, start_time, n_procs, print_tables=True):
    exp_directories = os.listdir(os.path.join(base_path, "log"))

    # Write a seperator so that we can easily identify the start of a new update
    if print_tables:
        print(f"\n\n\n\n\n\n\n\n{'-' * 20}{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{'-' * 20}\n")
    tot_completed_games = 0
    for exp_name in exp_directories:
        # May be an old experiment that has already been processed
        if exp_name not in tables:
            continue

        ai_stats = Counter()
        path_to_result, path_to_log = prepare_paths(base_path, exp_name)
        log_files = glob.glob(f"{path_to_log}/*.log")
        game_name = exp_name.split("_")[0]

        with open(os.path.join(path_to_result, f"{game_name}_results.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            if exp_name in tables:
                f.write(tables[exp_name]["description"])
            completed_games, error_games, draws = process_log_files(log_files, ai_stats, writer)
            tot_completed_games += completed_games
        # Write cumulative table to separate CSV file
        write_cumulative_table(path_to_result, game_name, tables, exp_name, ai_stats, completed_games)
        # Print tables, if required
        if print_tables:
            print_experiment_stats(tables, exp_name, ai_stats, completed_games, error_games, draws)

        print(
            f"\n{'- -'* 30}\n\nOverall completed/total games: {tot_completed_games}/{total_games}, errors: {error_games}\n"
        )
        if tot_completed_games > 0:
            games_remaining = total_games - tot_completed_games
            sets_remaining = math.ceil(games_remaining / n_procs)
            elapsed_time_seconds = (datetime.datetime.now() - start_time).seconds
            # Adjusting the average time per game calculation to account for parallel execution
            effective_average_time_per_game = elapsed_time_seconds / (tot_completed_games / n_procs)
            estimated_time_remaining_seconds = effective_average_time_per_game * sets_remaining

            print(f"Average time per game (adjusted for parallelism): {format_time(effective_average_time_per_game)}")
            print(f"Estimated time remaining: {format_time(estimated_time_remaining_seconds)}\n{'- -' * 30}\n")


def process_log_files(log_files, ai_stats, writer):
    completed_games, error_games, draws = 0, 0, 0

    def extract_winner_loser_info(log_contents):
        winner_search = re.search(r"\[(.*)\]", log_contents[-2])
        loser_info = log_contents[-3]
        loser_search = re.search(r"\[(.*)\]", loser_info)

        # Initialize default values in case the search fails
        winner_params = None
        loser_params = None

        # Check if the searches found matches before accessing groups
        if winner_search:
            winner_params = winner_search.group(1)
        if loser_search:
            loser_params = loser_search.group(1)

        # Return the extracted parameters, which might be None if not found
        return winner_params, loser_params

    for log_file in log_files:
        with open(log_file, "r") as log_f:
            log_contents = log_f.readlines()

        if len(log_contents) < 3:
            continue

        game_number = log_file.split("/")[-1].split(".")[0]
        if "Experiment completed" in log_contents[-1]:
            completed_games += 1
            winner_params, loser_params = extract_winner_loser_info(log_contents)
            update_stats_and_write(ai_stats, winner_params, loser_params, writer, game_number)
        elif "Experiment error" in log_contents[-1]:
            writer.writerow([game_number, "Error"])
            error_games += 1
        else:
            draws += 1
            writer.writerow([game_number, "Draw"])

    return completed_games, error_games, draws


def update_stats_and_write(ai_stats, winner_params, loser_params, writer, game_number):
    if winner_params not in ai_stats:
        ai_stats[winner_params] = 0
    if loser_params not in ai_stats:
        ai_stats[loser_params] = 0

    ai_stats[winner_params] += 1  # Update AI statistics
    writer.writerow([game_number, winner_params])


def write_cumulative_table(path_to_result, game_name, tables, exp_name, ai_stats, completed_games):
    with open(os.path.join(path_to_result, f"{game_name}_cumulative_stats.csv"), "w", newline="") as f:
        if exp_name in tables:
            f.write(tables[exp_name]["description"])
        writer = csv.writer(f)
        writer.writerow(["AI", "Win %", "95% C.I.", "# Games"])
        Z = 1.96
        for ai, wins in ai_stats.items():
            win_rate, ci_width = calculate_win_rate_and_ci(wins, completed_games, Z)
            writer.writerow([ai, f"{win_rate * 100:.2f}", f"±{ci_width * 100:.2f}", completed_games])


def calculate_win_rate_and_ci(wins, completed_games, Z):
    win_rate = wins / completed_games if completed_games > 0 else 0
    ci_width = Z * math.sqrt((win_rate * (1 - win_rate)) / completed_games) if completed_games > 0 else 0
    return win_rate, ci_width


def print_experiment_stats(tables, exp_name, ai_stats, completed_games, error_games, draws):

    # Initialize PrettyTable with headers
    print_stats = PrettyTable([f"({exp_name}) AI Name", "Win %", "95% C.I.", "Wins", "Total Games"])

    # Z-score for 95% confidence interval
    Z = 1.96

    # Populate the table with AI stats
    for ai, wins in ai_stats.items():
        win_rate, ci_width = calculate_win_rate_and_ci(wins, completed_games, Z)
        print_stats.add_row([ai, f"{win_rate * 100:.2f}%", f"±{ci_width * 100:.2f}%", wins, completed_games])

    # Print the statistics table for the current experiment
    print(f"Statistics for {exp_name}:")

    print(tables[exp_name]["description"])
    print(tables[exp_name]["start_time"])
    print(print_stats)
    tables[exp_name]["table"] = print_stats.get_string()

    # Add summary of games below the table
    summary = f"Completed Games: {completed_games}, Errors: {error_games}, Draws: {draws}"
    print(summary)


# def update_running_experiment_status(tables, base_path, print_tables=True) -> list[str]:
#     exp_directories = os.listdir(os.path.join(base_path, "log"))

#     for exp_name in exp_directories:
#         completed_games = 0
#         error_games = 0
#         draws = 0
#         ai_stats = Counter()  # To hold cumulative statistics per AI

#         path_to_result = os.path.join(base_path, "results", exp_name)
#         path_to_log = os.path.join(base_path, "log", exp_name)
#         os.makedirs(path_to_log, exist_ok=True)
#         os.makedirs(path_to_result, exist_ok=True)

#         # The first part of the exp_name is the name of the game
#         game_name = exp_name.split("_")[0]
#         log_files = glob.glob(f"{path_to_log}/*.log")
#         stuck_games_list = []

#         # Open CSV file in write mode (it needs to be overwritten every time)
#         with open(os.path.join(path_to_result, f"{game_name}_results.csv"), "w", newline="") as f:
#             if exp_name in tables:
#                 f.write(tables[exp_name]["description"])

#             writer = csv.writer(f)

#             for log_file in log_files:
#                 with open(log_file, "r") as log_f:
#                     log_contents = log_f.readlines()

#                     if len(log_contents) < 3:
#                         continue

#                     game_number = log_file.split("/")[-1].split(".")[0]  # Get game number from log file name
#                     # ! Assuming "Experiment completed" is second to last line, this must remain true
#                     if "Experiment completed" in log_contents[-1]:
#                         completed_games += 1
#                         # Assuming the last line is "Game Over. Winner: Px [px_params]"
#                         winner_info = log_contents[-2]
#                         winner_params = re.search(r"\[(.*)\]", winner_info).group(1)

#                         if "Game Over. Winner: P1" in winner_info or "Game Over. Winner: P2" in winner_info:
#                             loser_info = log_contents[-3]
#                             loser_params = re.search(r"\[(.*)\]", loser_info).group(1)

#                             # Make sure there's a line for each player
#                             if winner_params not in ai_stats:
#                                 ai_stats[winner_params] = 0
#                             if loser_params not in ai_stats:
#                                 ai_stats[loser_params] = 0

#                             writer.writerow([game_number, winner_params])
#                             ai_stats[winner_params] += 1  # Update AI statistics
#                         else:
#                             draws += 1
#                             writer.writerow([game_number, "Draw"])
#                     elif "Experiment error" in log_contents[-1]:  # Assuming "Experiment error" is the last line
#                         writer.writerow([game_number, "Error"])
#                         error_games += 1

#         # Write cumulative table to separate CSV file
#         with open(f"{path_to_result}/{game_name}_cumulative_stats.csv", "w", newline="") as f:
#             if exp_name in tables:
#                 f.write(tables[exp_name]["description"])
#             writer = csv.writer(f)
#             writer.writerow(["AI", "Win %", "95% C.I.", "# Games"])

#             Z = 1.96
#             for ai, wins in ai_stats.items():
#                 if completed_games > 0:
#                     win_rate = wins / completed_games
#                     ci_width = Z * math.sqrt((win_rate * (1 - win_rate)) / completed_games)
#                     writer.writerow(
#                         [
#                             ai,
#                             f"{win_rate * 100:.2f}",
#                             f"±{ci_width * 100:.2f}",
#                             completed_games,
#                         ]
#                     )
#                 else:
#                     writer.writerow([ai, "N/A", "N/A", 0])

#         if print_tables:
#             # Print cumulative statistics per AI to the screen
#             print_stats = PrettyTable(
#                 [
#                     f"AI ({exp_name})",
#                     f"Win % (Games: {completed_games}, Errors: {error_games}, Draws: {draws})",
#                     "95% C.I.",
#                 ]
#             )
#             # Z-score for 95% confidence interval
#             Z = 1.96
#             # Add rows to the table
#             for ai, wins in ai_stats.items():
#                 win_rate = wins / completed_games
#                 ci_width = Z * math.sqrt((win_rate * (1 - win_rate)) / completed_games)
#                 print_stats.add_row([ai, f"{win_rate * 100:.2f}", f"±{ci_width*100:.2f}"])

#             # Keep track of all experiments, also the finished ones to print
#             tables[exp_name]["table"] = print_stats
#             print("\n\n\n\n")
#             print("-" * 20, end="")
#             print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), end="")
#             print("-" * 20)
#             print("\n")
#             for _, v in tables.items():
#                 print(v["description"])
#                 print(v["table"])

#             if stuck_games_list != []:
#                 print(f"Stuck games: {stuck_games_list}")

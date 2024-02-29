from collections import Counter
import csv
import datetime
import glob
import itertools
import json
import math
import os
import re
import pandas as pd
from prettytable import PrettyTable


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
                            continue

                        # Parse metadata using regular expressions
                        try:
                            metadata["exp_name"] = re.search(r"exp_name='(.*?)'", lines[0]).group(1)
                            metadata["date_time"] = re.search(r"(\d{8}_\d{6})", metadata["exp_name"]).group(1)
                            metadata["game_name"] = re.search(r"game_name='(.*?)'", lines[0]).group(1)
                            metadata["game_params"] = re.search(r"game_params\s*=\s*(\{.*?\})", lines[1]).group(1)
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

            if len(aggregated_rows) > 0:
                print(f"Writing {len(aggregated_rows)} rows to {output_file}.")
                print(f"First row: {aggregated_rows[1]}")
                print(f"Last row: {aggregated_rows[-1]}")
                # Sort the aggregated rows by the AI1 win rate
                try:
                    aggregated_rows.sort(key=lambda row: float(row[8]), reverse=False)
                except ValueError as e:
                    print(f"Error sorting rows: {e}, {file}")

                # Write the sorted rows to the output file
                for row in aggregated_rows:
                    writer.writerow(row)
    except Exception as e:
        print(f"Error aggregating results: {e}, skipping {file}")
        import traceback

        # Print the stack trace so we can find the error
        traceback.print_exc()


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


def update_running_experiment_status(exp_name, tables, base_path, print_tables=True) -> list[str]:
    completed_games = 0
    error_games = 0
    draws = 0
    ai_stats = Counter()  # To hold cumulative statistics per AI

    path_to_result = os.path.join(base_path, "results", exp_name)
    path_to_log = os.path.join(base_path, "log", exp_name)
    os.makedirs(path_to_log, exist_ok=True)
    os.makedirs(path_to_result, exist_ok=True)
    # The first part of the exp_name is the name of the game
    game_name = exp_name.split("_")[0]
    log_files = glob.glob(f"{path_to_log}/*.log")
    stuck_games_list = []

    # Open CSV file in write mode (it needs to be overwritten every time)
    with open(os.path.join(path_to_result, f"{game_name}_results.csv"), "w", newline="") as f:
        if exp_name in tables:
            f.write(tables[exp_name]["description"])

        writer = csv.writer(f)

        for log_file in log_files:
            with open(log_file, "r") as log_f:
                log_contents = log_f.readlines()

                if len(log_contents) < 3:
                    continue

                game_number = log_file.split("/")[-1].split(".")[0]  # Get game number from log file name
                # ! Assuming "Experiment completed" is second to last line, this must remain true
                if "Experiment completed" in log_contents[-1]:
                    completed_games += 1
                    # Assuming the last line is "Game Over. Winner: Px [px_params]"
                    winner_info = log_contents[-2]
                    winner_params = re.search(r"\[(.*)\]", winner_info).group(1)

                    if "Game Over. Winner: P1" in winner_info or "Game Over. Winner: P2" in winner_info:
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
                elif "Experiment error" in log_contents[-1]:  # Assuming "Experiment error" is the last line
                    writer.writerow([game_number, "Error"])
                    error_games += 1

    # Write cumulative table to separate CSV file
    with open(f"{path_to_result}/{game_name}_cumulative_stats.csv", "w", newline="") as f:
        if exp_name in tables:
            f.write(tables[exp_name]["description"])
        writer = csv.writer(f)
        writer.writerow(["AI", "Win %", "95% C.I.", "# Games"])

        Z = 1.96
        for ai, wins in ai_stats.items():
            if completed_games > 0:
                win_rate = wins / completed_games
                ci_width = Z * math.sqrt((win_rate * (1 - win_rate)) / completed_games)
                writer.writerow(
                    [
                        ai,
                        f"{win_rate * 100:.2f}",
                        f"±{ci_width * 100:.2f}",
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
            print_stats.add_row([ai, f"{win_rate * 100:.2f}", f"±{ci_width*100:.2f}"])

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
                    print(f"Average time per game: {(v['end_time'] - v['start_time']) / float(completed_games)}")
                if stuck_games_list != []:
                    print(f"Stuck games: {stuck_games_list}")

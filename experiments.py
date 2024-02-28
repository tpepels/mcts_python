from copy import deepcopy
import datetime

from multiprocessing import Process

import os
import random
import shutil
import time

import traceback

from experiment_functions import (
    ColName,
    aggregate_csv_results,
    expand_rows,
    generate_ascii_art,
    update_running_experiment_status,
)

from run_games import AIParams, init_game, run_game_experiment
from util import redirect_print_to_log


tables = {}


def start_experiments_from_json(json_file_path, n_procs=4, count_only=False, agg_loc=None):
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
    random.shuffle(expanded_experiment_configs)
    # Step 2: Start experiments using multiprocessing
    for exp_dict in expanded_experiment_configs:
        processes, exp_name = run_new_experiment(exp_dict, 30 * 60)  # Adjusted to return processes

        # Wait for all processes to complete
        all_done = False
        while not all_done:
            time.sleep(30)  # Wait a bit before checking again
            # Update running experiment status periodically
            update_running_experiment_status(exp_name=exp_name, tables=tables, base_path=base_path)

            # Check if all processes are done
            all_done = all(process.is_alive() == False for process in processes)

        tables[exp_name]["end_time"] = datetime.datetime.now()

        # Post-experiment updates
        time.sleep(30)
        update_running_experiment_status(exp_name=exp_name, tables=tables, base_path=base_path)
        if agg_loc is not None:
            aggregate_csv_results(agg_loc, base_path)


def run_new_experiment(exp_dict, timeout=30 * 60):
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
    games_params = [game for game in games_params if game[0] >= start_game]  # Make sure the process has ended

    processes = []
    for exp_params in games_params:

        p = Process(target=experiment_wrapper, args=exp_params, kwargs={"timeout": timeout})
        p.start()
        processes.append(p)

    """
    Write the experiment configuration as a header to a CSV file in the log directory.
    So we can easily find results of a specific experiment.
    """
    path_to_result = os.path.join(base_path, "results", exp_name)
    path_to_log = os.path.join(base_path, "log", exp_name)
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

    return processes, exp_name


def experiment_wrapper(*args, timeout=30 * 60):
    # Start the actual experiment in a separate process
    p = Process(target=run_single_experiment, args=args)

    p.start()
    p.join(timeout)  # Wait for the process to complete or timeout

    if p.is_alive():
        p.terminate()  # Terminate the process if it is still alive
        p.join()


def run_single_experiment(
    i: int,
    game_key: str,
    game_params: dict,
    p1_params: AIParams,
    p2_params: AIParams,
    exp_name: str,
    base_path: str = ".",
    random_openings: int = 0,
) -> None:

    log_path = os.path.join(base_path, "log", exp_name, f"{i}.log")
    try:
        with redirect_print_to_log(log_path):
            run_game_experiment(game_key, game_params, p1_params, p2_params, random_openings)

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
    import argparse

    parser = argparse.ArgumentParser(description="Start experiments based on JSON config.")
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
        parser.error("Either --json_file should be set OR --aggregate_resultsshould be enabled.")

    global base_path

    if args.aggregate_results and not args.json_file:
        base_path = args.base_path
        # If no json file was given, just aggregate the results
        print(f"Aggregating results from {base_path} to {args.aggregate_results}")
        aggregate_csv_results(args.aggregate_results, base_path)
        return

    # Include the experiment file in the base_path
    base_path = os.path.join(args.base_path, os.path.splitext(os.path.basename(args.json_file))[0])
    print("Base path:", base_path)
    # Use the name of the JSON file with a .csv extension if --aggregate_results is not provided.
    agg_loc = os.path.join(base_path, os.path.splitext(os.path.basename(args.json_file))[0] + ".csv")

    if not args.count_only:
        # Validate and create the base directory for logs if it doesn't exist
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        elif args.clean:
            # Check if agg_loc file exists and rename it by appending a timestamp
            if os.path.exists(agg_loc):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = os.path.join(
                    base_path,
                    os.path.splitext(os.path.basename(args.json_file))[0] + f"_{timestamp}.csv",
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
    import json

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
    aggregate_csv_results(agg_loc, base_path)

    text = "Experiment Finished!"
    generate_ascii_art(text)


if __name__ == "__main__":
    main()

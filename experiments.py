from concurrent.futures import CancelledError
from copy import deepcopy
import datetime
import json
from pprint import pprint

import sys
from pebble import ProcessFuture, ProcessPool, ProcessExpired
import threading
import time

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
from split_params import do_split
from util import redirect_print_to_log


tables = {}


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
            print("Running experiment", i)
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


# This has the form "exp_name": False/True
# True means the experiment is running and should be cancelled, False means it has been cancelled
experiments_to_cancel = {}


def start_experiments_from_json(json_file_path, n_procs=8, count_only=False, agg_loc=None, timeout=20 * 60):
    with open(json_file_path) as json_file:
        data = json.load(json_file)

    print(f"Running using {n_procs} processes.")

    # Check if the first element contains the 'top_n' configuration
    top_n = 0  # Default value indicating no top_n filtering
    if "top_n" in data[0]:
        top_n = data[0]["top_n"]
        print("Top_n filtering enabled: ", top_n)
        expanded_experiment_configs = expand_rows(data[1:])  # Exclude the top_n configuration
    else:
        expanded_experiment_configs = expand_rows(data)

    print(f"{len(expanded_experiment_configs)} experiments.")
    if count_only:
        return

    # Interval for status updates
    update_interval = 240
    stop_event = threading.Event()

    # Process experiments
    all_experiments = [experiment for config in expanded_experiment_configs for experiment in get_experiments(config)]
    # Start the periodic status update thread (tables and base_path are global variables)
    total_games = len(all_experiments)
    status_thread = threading.Thread(
        target=run_periodic_status_updates,
        args=(update_interval, stop_event, tables, base_path, total_games, n_procs, agg_loc, top_n),
    )
    status_thread.start()
    random.shuffle(all_experiments)
    total_cancelled = 0
    with ProcessPool(max_workers=n_procs) as pool:
        print(f"Starting {len(all_experiments)} experiments with {n_procs} processes.")
        future_tasks = {}
        futures_list = []
        for exp in all_experiments:
            exp_name = exp[-3]
            if exp_name not in future_tasks:
                future_tasks[exp_name] = []

            # Schedule the task and store the future with the exp_name
            future = pool.schedule(run_single_experiment, args=exp)
            future_tasks[exp_name].append(future)
            futures_list.append(future)

        print(f"All {len(all_experiments)} experiments pooled.")
        future: ProcessFuture
        for future in futures_list:
            try:
                if not future.done():
                    # Blocks until the task completes or raises TimeoutError if it times out
                    future.result(timeout=timeout)
            except CancelledError:
                print("A task was cancelled and did not complete.")
            except TimeoutError as e:
                print(f"Experiment timed out: {e}", sys.stderr)
            except ProcessExpired as e:
                print(f"Experiment process expired: {e}", sys.stderr)
            except Exception as e:
                print(f"Experiment raised an exception: {e}", sys.stderr)
                # Print the traceback
                traceback.print_exc(file=sys.stderr)

            # Everytime we finish a task, check if some experiments should be cancelled
            exp_names_to_cancel = [
                exp_name for exp_name, should_cancel in experiments_to_cancel.items() if should_cancel
            ]

            if len(exp_names_to_cancel) > 0:
                print("Experiments to cancel:")
                pprint(experiments_to_cancel)
                total_cancelled += cancel_list_of_experiments(future_tasks, exp_names_to_cancel)

            if total_cancelled > 0:
                print(f"Cancelled {total_cancelled} tasks so far.")
                print("Cancelled experiments:")
                cancelled_list_of_experiments = [
                    exp_name for exp_name, should_cancel in experiments_to_cancel.items() if not should_cancel
                ]
                print(cancelled_list_of_experiments)

            # Check if all futures are done or cancelled
            if all((future.done() or future.cancelled()) for future in futures_list):
                print("All futures completed.")
                break

        print("--" * 60)
        print("All futures completed.")
        print("--" * 60)

    if agg_loc is not None:
        print("Aggregating final results..")
        aggregate_csv_results(agg_loc, base_path)

    # Experiments are done, signal the status update thread to stop and wait for it to finish
    stop_event.set()
    print("Waiting for status update thread to finish..")
    status_thread.join(timeout=10)

    # perform any final status update after all experiments are complete
    update_running_experiment_status(
        tables=tables,
        base_path=base_path,
        total_games=total_games,
        start_time=datetime.datetime.now(),
        n_procs=n_procs,
    )
    print("Completed all experiments and updates.")


def cancel_list_of_experiments(future_tasks, experiment_names_to_cancel):
    total_cancelled = 0
    for exp_name in experiment_names_to_cancel:
        if exp_name in future_tasks:
            count = 0
            for future in future_tasks[exp_name]:
                if not future.done():
                    future.cancel()
                    count += 1
            total_cancelled += count
            print(f"Cancelled {count} tasks for {exp_name}.")
            # Mark the experiment as cancelled
            experiments_to_cancel[exp_name] = False
    return total_cancelled


def run_periodic_status_updates(
    update_interval, stop_event, tables_dict, base_path, total_games, n_procs, agg_loc=None, top_n=0
):
    """
    Periodically runs the update_running_experiment_status function until a stop event is set.

    Args:
        update_interval (int): The interval between updates in seconds.
        stop_event (threading.Event): An event to signal the thread to stop.
        tables (dict): The shared data structure to pass to the update function.
        base_path (str): The base path to pass to the update function.
    """
    start_time = datetime.datetime.now()
    counter = 0
    while not stop_event.is_set():
        time.sleep(update_interval // 2)
        print("Updating status..")
        update_running_experiment_status(
            tables_dict, base_path=base_path, total_games=total_games, start_time=start_time, n_procs=n_procs
        )
        print(f"Status update complete, {datetime.datetime.now()}.\n")
        time.sleep(update_interval // 2)
        counter += 1
        # Every n updates, aggregate the results and print them
        if counter % 2 == 0 and agg_loc is not None:
            print("aggregating results..")
            try:
                cancel_list = aggregate_csv_results(agg_loc, base_path, top_n=top_n)

                # Update the global cancelled_experiments dictionary based on cancel_list
                for exp_name in cancel_list:
                    # Mark for cancellation if not already cancelled
                    if exp_name not in experiments_to_cancel:
                        experiments_to_cancel[exp_name] = True
                        print(f"Marked {exp_name} for cancellation.")

                print(f"Aggregation complete, {datetime.datetime.now()}.\n")
            except Exception as e:
                # This is probably because the file is still being written to or some such reason
                print(f"Error aggregating results: {e}", sys.stderr)
                # print Traceback
                traceback.print_exc(file=sys.stderr)


exp_names = []


def get_experiments(exp_dict):
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

    exp_name = f"{game}_{exp_dict[ColName.P1_AI_KEY]}_{exp_dict[ColName.P2_AI_KEY]}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    if exp_name in exp_names:
        exp_name += f"_{len(exp_names)}"

    exp_names.append(exp_name)

    start_game = 0

    del game
    n_games = exp_dict[ColName.N_GAMES]
    print(f"Expanding experiment {exp_name}")

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

    """
    Write the experiment configuration as a header to a CSV file in the log directory.
    So we can easily find results of a specific experiment.
    """
    path_to_result = os.path.join(base_path, "results", exp_name)
    path_to_log = os.path.join(base_path, "log", exp_name)
    os.makedirs(path_to_log, exist_ok=True)
    os.makedirs(path_to_result, exist_ok=True)
    global tables
    tables[exp_name] = {}
    description_str = f"{exp_name=}  --  {game_name=}\n"
    description_str += f"{game_params=}\n"
    description_str += f"{p1_params=}\n"
    description_str += f"{p2_params=}\n"
    description_str += f"{n_games=}\n"
    description_str += f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n     -------- \n"
    tables[exp_name]["description"] = description_str
    tables[exp_name]["start_time"] = datetime.datetime.now()

    return games_params


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Start experiments based on JSON config.")

    parser.add_argument(
        "--split_file", help="The JSON file containing the experiment configurations to split.", default=None, type=str
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

    if args.split_file:
        print(f"Splitting experiments from {args.split_file}")
        do_split(args.split_file)
        return

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

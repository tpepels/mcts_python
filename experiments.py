import csv
import datetime
import glob
import itertools
import json
import multiprocessing
import os
import random
import re
import socket
import time
import traceback
from collections import Counter
from multiprocessing.pool import AsyncResult
from typing import Any

import gspread
import numpy as np
import pandas as pd
from gspread_dataframe import set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials
from prettytable import PrettyTable
from termcolor import colored

from run_games import AIParams, init_game, run_game_experiment
from util import read_config, redirect_print_to_log

# USE THE FOLLOWING AS HEADER FOR THE GOOGLE SHEET
# status, n_games, start, completed_games, errors, worksheet, game_key, game_params, p1_ai_key, p1_ai_params, p1_eval_key, p1_eval_params, p2_ai_key, p2_ai_params, p2_eval_key, p2_eval_params

WRITE_TO_SHEET = True


class ColName:
    STATUS = "status"
    N_GAMES = "n_games"
    START = "start"
    COMPLETED_GAMES = "completed_games"
    ERROR_GAMES = "n_errors"
    WORKSHEET = "worksheet"
    ERROR = "errors"
    GAME_KEY = "game_key"
    GAME_PARAMS = "game_params"
    P1_AI_KEY = "p1_ai_key"
    P1_AI_PARAMS = "p1_ai_params"
    P1_EVAL_KEY = "p1_eval_key"
    P1_EVAL_PARAMS = "p1_eval_params"
    P2_AI_KEY = "p2_ai_key"
    P2_AI_PARAMS = "p2_ai_params"
    P2_EVAL_KEY = "p2_eval_key"
    P2_EVAL_PARAMS = "p2_eval_params"


class ColIndex:
    # Column indices in Google Sheets (1-based)
    STATUS = 1
    N_GAMES = 2
    START = 3
    COMPLETED_GAMES = 4
    ERROR_GAMES = 5
    WORKSHEET = 6
    ERROR_MESSAGE = 7


class Status:
    # Status values
    PENDING = "Pending"
    RUNNING = "Running"
    COMPLETED = "Completed"
    INTERRUPTED = "Interrupted"
    RESUME = "Resume"
    ERROR = "Error"


def mark_experiment_as_status(
    sheet: gspread.Worksheet,
    index,
    status,
    n_games=None,
    start_time=None,
    completed_games=0,
    error_games=0,
    worksheet_name=None,
    error_message=None,
):
    """
    Update the status of an experiment in the Google Sheets document. If the status is 'Running',
    it also updates the total number of games, the start time, the number of completed games, and
    the number of games with errors. It can also update the error message if provided.

    Args:
        sheet (gspread.Worksheet): The Google Sheets document.
        index (int): The index of the experiment in the DataFrame (0-based).
        status (str): The new status of the experiment ('Pending', 'Running', or 'Completed').
        n_games (int, optional): The total number of games for the experiment. Set this parameter only once at the beginning of the experiment.
        start_time (datetime, optional): The start time of the experiment. If not provided and the status is 'Running',
                                         the current time is used.
        completed_games (int, optional): The number of completed games.
        error_games (int, optional): The number of games with errors.
        error_message (str, optional): The error message to be written in the sheet.

    Raises:
        Exception: Any error occurred during the update operation.

    Returns:
        None
    """
    try:
        # We add 2 because Google Sheets has 1-based indexing and the first row is the header
        sheet.update_cell(index + 2, ColIndex.STATUS, status)
        if status in [Status.RUNNING, Status.COMPLETED]:
            if n_games:
                sheet.update_cell(index + 2, ColIndex.N_GAMES, n_games)  # Total number of games
            if start_time:
                sheet.update_cell(
                    index + 2, ColIndex.START, start_time.strftime("%Y-%m-%d %H:%M:%S")
                )  # Start time
            if completed_games > 0:
                sheet.update_cell(index + 2, ColIndex.COMPLETED_GAMES, completed_games)  # Completed games
            if error_games > 0:
                sheet.update_cell(index + 2, ColIndex.ERROR_GAMES, error_games)  # Games with errors
            if worksheet_name:
                sheet.update_cell(index + 2, ColIndex.WORKSHEET, worksheet_name)
        if error_message:  # If there's an error message, update the error message column
            sheet.update_cell(index + 2, ColIndex.ERROR_MESSAGE, error_message)
    except Exception as e:
        print(f"Failed to mark experiment as {status}: {e}")


def get_experiments_from_sheet(sheet: gspread.Worksheet):
    """
    Read the experiment data from a Google Sheets document, filter out the experiments that are not 'Pending',
    parse their parameters, and expand the rows for parameter combinations. The modified DataFrame is written back to
    the Google Sheets document only if any changes have been made due to the expansion of rows.

    Args:
        sheet (gspread.Worksheet): The Google Sheets document.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the experiment data.

    Example:
        Assuming that the Google Sheets document has the following content:
        | status | game_params             | ai_params            |
        |--------|-------------------------|----------------------|
        | Pending| {"param1": "[1, 2]"}    | {"param2": "3:5:1"}  |
        | Running| {"param3": "[4, 5]"}    | {"param4": "6:8:1"}  |

        get_experiments_from_sheet(sheet) would return a DataFrame as follows:

        | status | game_params  | ai_params   |
        |--------|--------------|-------------|
        | Pending| {"param1": 1}| {"param2": 3|
        | Pending| {"param1": 2}| {"param2": 3|
        | Pending| {"param1": 1}| {"param2": 4|
        | Pending| {"param1": 2}| {"param2": 4|
    """
    data = sheet.get_all_records()

    # If sheet is empty, write headers and create empty dataframe
    if not data:
        headers = [
            ColName.STATUS,
            ColName.N_GAMES,
            ColName.START,
            ColName.COMPLETED_GAMES,
            ColName.ERROR_GAMES,
            ColName.WORKSHEET,
            ColName.ERROR,
            ColName.GAME_KEY,
            ColName.GAME_PARAMS,
            ColName.P1_AI_KEY,
            ColName.P1_AI_PARAMS,
            ColName.P1_EVAL_KEY,
            ColName.P1_EVAL_PARAMS,
            ColName.P2_AI_KEY,
            ColName.P2_AI_PARAMS,
            ColName.P2_EVAL_KEY,
            ColName.P2_EVAL_PARAMS,
        ]
        # Check if the sheet already has headers
        if not sheet.row_values(1):
            sheet.append_row(headers)

        df = pd.DataFrame(columns=headers)
        return df
    else:
        df = pd.DataFrame(data)

    # Save the original dataframe before applying the filter
    original_df = df.copy()

    # Only process experiments that are 'Pending'
    df = df[df[ColName.STATUS] == Status.PENDING]

    df[ColName.GAME_PARAMS] = df.apply(lambda row: parse_parameters(row[ColName.GAME_PARAMS]), axis=1)
    df[ColName.P1_AI_PARAMS] = df.apply(lambda row: parse_parameters(row[ColName.P1_AI_PARAMS]), axis=1)
    df[ColName.P2_AI_PARAMS] = df.apply(lambda row: parse_parameters(row[ColName.P2_AI_PARAMS]), axis=1)
    df[ColName.P1_EVAL_PARAMS] = df.apply(lambda row: parse_parameters(row[ColName.P1_EVAL_PARAMS]), axis=1)
    df[ColName.P2_EVAL_PARAMS] = df.apply(lambda row: parse_parameters(row[ColName.P2_EVAL_PARAMS]), axis=1)

    # Expand rows for parameter combinations
    df_expanded = expand_rows(df)

    # Merge the filtered/expanded dataframe with the original dataframe
    df = pd.concat(
        [original_df[original_df[ColName.STATUS] != Status.PENDING], df_expanded]
    ).drop_duplicates()

    temp_sheet = sheet.spreadsheet.add_worksheet(title="temp", rows="100", cols="20")
    # Write the df to the temporary sheet
    set_with_dataframe(temp_sheet, df)

    # If everything has worked so far, copy the data to the main sheet and clear the temporary sheet
    sheet.clear()
    # Set the DataFrame to the main sheet, replacing the current contents
    set_with_dataframe(sheet, df)

    sheet.spreadsheet.del_worksheet(temp_sheet)
    # Convert json strings to dicts
    for col in [
        ColName.GAME_PARAMS,
        ColName.P1_AI_PARAMS,
        ColName.P2_AI_PARAMS,
        ColName.P1_EVAL_PARAMS,
        ColName.P2_EVAL_PARAMS,
    ]:
        df[col] = df[col].apply(lambda x: json.loads(x))

    return df


def parse_parameters(param_string: str):
    """
    Parse a string containing parameter details into a dictionary. If a value is a JSON list or a range (in the format
    'start:end:step'), it's converted into a Python list.

    Args:
        param_string (str): A string containing parameter details. Expected to be a JSON-like string.
        sheet (gspread.Worksheet): The Google Sheets document.
        index (int): The index of the experiment in the DataFrame (0-based).

    Returns:
        dict: A dictionary containing the parsed parameters.

    Example:
        param_string = '{"param1": "[1, 2]", "param2": "3:5:1"}'
        parse_parameters(param_string, sheet, 0) would return:
        {"param1": [1, 2], "param2": [3.0, 4.0]}
    """
    if not param_string:
        return None

    try:
        params = json.loads(param_string.replace("'", '"'))
        for key, value in params.items():
            if isinstance(value, str) and re.match(r"\[(.*)\]", value):  # Check if the value is a list
                params[key] = json.loads(value.replace("'", '"'))  # Converts the string to list using json
            elif isinstance(value, str) and ":" in value:  # Check if the value is a range
                start, end, step = map(float, value.split(":"))
                params[key] = list(np.arange(start, end, step))
        return params

    except Exception as e:
        print(f"Failed to parse parameters: {param_string}: {e}")
        return None


def expand_rows(df):
    # Find columns that need to be expanded
    params_cols = [col for col in df.columns if col.endswith("_params") and df[col].dtype == "object"]

    res = []
    for index, row in df.iterrows():
        # print(f"\nProcessing row {index}...")

        # If the parameters column is a dictionary, keep as is, else convert to an empty dictionary
        params_values = [
            (json.loads(row[col]) if isinstance(row[col], str) else row[col]) for col in params_cols
        ]
        # print(f"params_values before cleaning: {params_values}")

        # Handle None and other non-dictionary entries
        params_values = [param if isinstance(param, dict) else {} for param in params_values]
        # print(f"params_values after cleaning: {params_values}")

        # For each dictionary, create combinations of key and individual values
        keys_values_list = [
            [(col, *item) for item in list(itertools.product([k], v if isinstance(v, list) else [v]))]
            for col, dict_ in zip(params_cols, params_values)
            for k, v in dict_.items()
        ]
        # print(f"keys_values_list: {keys_values_list}")

        # Generate combinations of those combinations
        params_combinations = list(itertools.product(*keys_values_list))
        # print(f"params_combinations: {params_combinations}")

        # Convert each combination to a dictionary
        params_combinations_dicts = []
        for comb in params_combinations:
            params_dict = {}
            for col, key, value in comb:
                if col not in params_dict:
                    params_dict[col] = {}
                params_dict[col][key] = value
            params_combinations_dicts.append(params_dict)
        # print(f"params_combinations as dicts: {params_combinations_dicts}")

        # Add the new rows to the result
        for params_comb in params_combinations_dicts:
            new_row = row.copy()
            for col in params_cols:
                if col in params_comb:
                    new_row[col] = json.dumps(params_comb[col])
                else:
                    new_row[col] = json.dumps({})
            # print(f"Appending new_row: {new_row}")
            res.append(new_row)

    # Return as a DataFrame
    return pd.DataFrame(res)


def authorize_with_google_sheets():
    # Use credentials to create a client to interact with the Google Drive API
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("client_secret.json", scopes)
    client = gspread.authorize(creds)
    return client


def run_experiment(sheet, row, index, n_procs):
    try:
        print(colored(f"Starting new experiment at index {index}...", "yellow"))
        pool, async_result, sheet_name = run_new_experiment(row, index, sheet, n_procs)
        return pool, async_result, index, sheet_name
    except Exception as e:
        error_message = f"Error: {str(e)}"
        print(error_message)
        mark_experiment_as_status(sheet, index, Status.ERROR, error_message=error_message)
        return None, None, None, None


def monitor_sheet_and_run_experiments(interval: int, n_procs: int):
    client = authorize_with_google_sheets()

    print(colored("Getting Google Sheets configuration...", "yellow"))
    sheet = get_config_sheet(client)
    print(colored("Configuration fetched.", "green"))

    pool = None
    async_result = None
    current_experiment_index = None
    current_sheet_name = None
    try:
        while True:
            print(colored("Fetching experiments from Google Sheets...", "yellow"))
            df = get_experiments_from_sheet(sheet)
            print(colored(f"Fetched {df.shape[0]} experiments.", "green"))

            try:
                if is_experiment_running(async_result):
                    update_running_experiment_status(
                        sheet_name=current_sheet_name,
                        index=current_experiment_index,
                        sheet=sheet,
                        is_running=True,
                    )

                else:
                    # This probably means that the current experiment finished running
                    if async_result and async_result.ready():
                        print(colored(f"Experiment at index {current_experiment_index} completed.", "green"))
                        update_running_experiment_status(
                            sheet_name=current_sheet_name,
                            index=current_experiment_index,
                            sheet=sheet,
                            is_running=False,
                        )
                        mark_experiment_as_status(
                            current_sheet_name, current_experiment_index, Status.COMPLETED
                        )

                        pool = None
                        async_result = None
                        current_experiment_index = None

                    # Time to start a new experiment!
                    for index, row in df.iterrows():
                        if row[ColName.STATUS] in [Status.PENDING, Status.RESUME]:
                            pool, async_result, current_experiment_index, current_sheet_name = run_experiment(
                                sheet, row, index, n_procs
                            )
                            if current_experiment_index is not None:
                                break

                print(colored("Sleeping before next check...", "yellow"))
                time.sleep(interval)
                print("Pool:")
                print(f"    Number of Processes: {pool._processes}")
                print(f"    Current Task Queue: {pool._taskqueue}")
                print(f"    Current Workers: {pool._pool}")
                print(f"    Is Pool Terminated?: {pool._state}")

                print("Async Result:")
                print(f"    Ready?: {async_result.ready()}")
                if async_result.ready():
                    print(f"    Successful?: {async_result.successful()}")

            except KeyboardInterrupt:
                print(
                    colored(
                        "Interrupted. Attempting to mark currently running experiment as interrupted.", "red"
                    )
                )
                if is_experiment_running(async_result):
                    mark_experiment_as_status(sheet, current_experiment_index, Status.INTERRUPTED)
                break
            except Exception as e:
                print(colored(f"Exception {str(e)}! {traceback.format_exc()}", "red"))
                if is_experiment_running(async_result):
                    mark_experiment_as_status(
                        sheet, current_experiment_index, Status.ERROR, error_message=str(e)
                    )
                break
    finally:
        if current_experiment_index is not None:
            try:
                print(colored("Stopped... Processing final results.", "yellow"))
                update_running_experiment_status(
                    sheet_name=current_sheet_name,
                    index=current_experiment_index,
                    sheet=sheet,
                    is_running=False,
                )
            except Exception as ex:
                print(colored(f"Could not write final result to file/sheet {str(ex)}", "red"))

        if is_experiment_running(async_result):
            print(colored("Terminating running experiment...", "red"))
            pool.terminate()

        print(colored("Stopped monitoring.", "red"))


def is_experiment_running(async_result: AsyncResult):
    """
    Check if there is a running experiment process.

    This function checks whether the given process is currently running.

    Args:
        async_result (multiprocessing.Process): The process to check.

    Returns:
        bool: True if the process is running, False otherwise.
    """
    return async_result and not async_result.ready()


def run_new_experiment(row: pd.Series, index: int, sheet: gspread.Worksheet, n_procs: int):
    """
    Prepare and run a new experiment.

    This function prepares the parameters needed to run a new experiment based on
    information in a given row of a DataFrame. It then starts a new process to run
    the experiment, and updates the status of the experiment in the Google Sheets document.

    Args:
        row (pandas.Series): A row from the DataFrame representing the experiments table.
        index (int): The index of the row in the DataFrame.
        sheet (gspread.Worksheet): The worksheet of the Google Sheets document.

    Returns:
        multiprocessing.Process: The process that was started to run the experiment.
    """
    try:
        game_params = row[ColName.GAME_PARAMS]
        game_name = row[ColName.GAME_KEY]
        game = init_game(game_name, game_params=game_params)

        p1_params = AIParams(
            ai_key=row[ColName.P1_AI_KEY],
            ai_params=row[ColName.P1_AI_PARAMS],
            max_player=1,
            eval_key=row[ColName.P1_EVAL_KEY],
            eval_params=row[ColName.P1_EVAL_PARAMS],
            transposition_table_size=game.transposition_table_size,
        )
        p2_params = AIParams(
            ai_key=row[ColName.P2_AI_KEY],
            ai_params=row[ColName.P2_AI_PARAMS],
            max_player=2,
            eval_key=row[ColName.P2_EVAL_KEY],
            eval_params=row[ColName.P2_EVAL_PARAMS],
            transposition_table_size=game.transposition_table_size,
        )

        if row[ColName.STATUS] == Status.PENDING:
            sheet_name = f"{game}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            start_game = 0
        else:
            sheet_name = row[ColName.WORKSHEET]
            if row[ColName.COMPLETED_GAMES]:
                start_game = int(row[ColName.COMPLETED_GAMES])
            else:
                start_game = 0

            print(f"Restarting pending experiment from game {start_game}")

        del game

        n_games = row[ColName.N_GAMES]
        print(f"starting experiment {sheet_name}")

        games_params = [
            (game_name, game_params, p1_params, p2_params, False, sheet_name, WRITE_TO_SHEET)
            if i < n_games / 2
            else (game_name, game_params, p2_params, p1_params, True, sheet_name, WRITE_TO_SHEET)
            for i in range(n_games)
        ]

        random.shuffle(games_params)
        games_params = [(i, *params) for i, params in enumerate(games_params)]
        games_params = [game for game in games_params if game[0] >= start_game]

        create_sheet_if_not_exists(sheet_name)

        pool = multiprocessing.Pool(n_procs)
        async_result = pool.starmap_async(run_single_experiment, games_params)

        time.sleep(5)

        if not async_result.ready():
            mark_experiment_as_status(
                sheet,
                index,
                Status.RUNNING,
                row[ColName.N_GAMES],
                datetime.datetime.now(),
                0,
                0,
                sheet_name,
            )
        else:
            print("Experiment ended before statuscould be updated.")

        return pool, async_result, sheet_name

    except Exception as e:
        error_message = f"Error: {str(e)}"
        print(error_message)
        mark_experiment_as_status(sheet, index, Status.ERROR, error_message=error_message)
        return None, None


def get_config_sheet(client: gspread.Client):
    """
    Fetch the Google Sheet specified in the configuration file.

    This function reads a config file to obtain the ID of the Google Sheet
    to be used for experiments, then uses the provided gspread client to open that sheet.

    Args:
        client (gspread.Client): A client for interacting with the Google Sheets API.

    Returns:
        gspread.models.Worksheet: The worksheet of the Google Sheets document with the hostname title.
    """
    # Read the config file
    config = read_config()
    sheet_id = config.get("Sheets", "ExperimentListID")

    # Open the Google Sheets document
    sheet = client.open_by_key(sheet_id)

    # Get the system hostname
    hostname = socket.gethostname()

    try:
        # Try to open the worksheet with the system hostname
        tab = sheet.worksheet(hostname)
    except gspread.WorksheetNotFound:
        # If the worksheet does not exist, create it
        tab = sheet.add_worksheet(title=hostname, rows="200", cols="20")

    return tab


def update_running_experiment_status(sheet_name, index: int, sheet: gspread.Worksheet, is_running: bool):
    """
    Scan log files for status updates and update Google Sheet.

    This function scans the log files associated with a given experiment for status
    updates. It then updates the status of the experiment in the Google Sheets document
    based on the contents of the log files.

    Args:

        index (int): The index of the row in the DataFrame.
        sheet (gspread.Worksheet): The worksheet of the Google Sheets document.
        is_running (bool): True if the experiment is still running, false otherwise.
    """
    completed_games = 0
    error_games = 0
    ai_stats = Counter()  # To hold cumulative statistics per AI
    os.makedirs(f"log/games/{sheet_name}/", exist_ok=True)
    log_files = glob.glob(f"log/games/{sheet_name}/?.log")

    # Open CSV file in append mode
    with open(f"log/games/{sheet_name}/_results.csv", "w", newline="") as f:
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
                        writer.writerow([sheet_name, game_number, winner_params])
                        ai_stats[winner_params] += 1  # Update AI statistics
                    else:
                        writer.writerow([sheet_name, game_number, "Draw"])
                elif "Experiment error" in log_contents[-1]:  # Assuming "Experiment error" is the last line
                    error_games += 1
                    writer.writerow([sheet_name, game_number, "Error"])

    # Print cumulative statistics per AI to the screen
    print_stats = PrettyTable(["AI", f"Win % (Games: {completed_games})"])
    for ai, wins in ai_stats.items():
        print_stats.add_row([ai, f"{(wins / completed_games) * 100: .2f}"])

    print(f"{sheet_name} - {completed_games} games completed.")
    print("***-" * 20)
    print(print_stats)

    if is_running:
        mark_experiment_as_status(sheet, index, Status.RUNNING, None, None, completed_games, error_games)
    else:
        mark_experiment_as_status(sheet, index, Status.COMPLETED, None, None, completed_games, error_games)


def run_single_experiment(
    i: int,
    game_key: str,
    game_params: dict[str, Any],
    p1_params: AIParams,
    p2_params: AIParams,
    players_switched: bool,
    worksheet_name: str,
    write_to_sheet: bool = True,
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
        with redirect_print_to_log(f"log/games/{worksheet_name}/{i}.log"):
            setup, total_time, avg_time_per_move, n_moves, result = run_game_experiment(
                game_key, game_params, p1_params, p2_params
            )

        with open(f"log/games/{worksheet_name}/{i}.log", "a") as log_file:
            # Write a status message to the log file
            log_file.write("Experiment completed")

    except Exception as e:
        with open(f"log/games/{worksheet_name}/{i}.log", "a") as log_file:
            log_file.write(f"Experiment error: {e}")

    avg_time_p1, avg_time_p2 = avg_time_per_move

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

    try:
        if write_to_sheet:
            # Retrieve the pre-created Google Sheets document
            worksheet = create_sheet_if_not_exists(worksheet_name)
            worksheet.append_row(
                [
                    i + 1,
                    setup,
                    total_time,
                    n_moves,
                    avg_time_p1,
                    avg_time_p2,
                    str(p1_params),
                    str(p2_params),
                    p1_result,
                    p2_result,
                    players_switched,
                ]
            )
    except Exception as e:
        print(f"An error occurred while writing to the sheet: {e}")
        with open(f"log/games/{worksheet_name}/{i}.log", "a") as log_file:
            log_file.write(f"Experiment error: {e}")


def create_sheet_if_not_exists(worksheet_name: str):
    if not WRITE_TO_SHEET:
        return

    client = authorize_with_google_sheets()
    # Get list of all spreadsheets
    spreadsheet_list = client.list_spreadsheet_files()

    # Check if worksheet exists in the list
    worksheet_exists = any(sheet["name"] == worksheet_name for sheet in spreadsheet_list)

    if not worksheet_exists:
        # Create a new worksheet
        main_sheet = client.create(worksheet_name)
        worksheet = main_sheet.get_worksheet(0)
        config = read_config()
        main_sheet.share(config["Share"]["GoogleAccount"], perm_type="user", role="writer")
        # Write headers
        worksheet.insert_row(
            [
                "Experiment",
                "Setup",
                "Total Time",
                "# Moves Made",
                "Avg Time Per Move p1",
                "Avg Time Per Move p2",
                "Player 1",
                "Player 2",
                "P1 Result",
                "P2 Result",
                "Swapped Seats",
            ],
            1,
        )
        # Write formulas to keep track of the results per AI (not per seat)
        worksheet.update_acell("L1", "Winrate (per AI)")
        worksheet.update_acell("L2", '=AVERAGEIF(I2:I, "1")')
        worksheet.update_acell("M1", "95% CI")
        worksheet.update_acell("M2", "=CONFIDENCE.T(0.05, STDEV(I2:I), COUNTA(I2:I))")
        worksheet.update_acell("N1", "Average Time per Move")
        worksheet.update_acell("N2", "=AVERAGE(E2:E, F2:F)")
        worksheet.update_acell("O1", "Average # of moves")
        worksheet.update_acell("O2", "=AVERAGE(D2:D)")
    else:
        # If the worksheet exists, open it
        main_sheet = client.open(worksheet_name)
        worksheet = main_sheet.get_worksheet(0)

    return worksheet


import argparse

if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="Monitor a Google Sheets document for new experiments and execute them."
    )

    # Add the interval argument
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        required=True,
        help="The interval in seconds between each check for new experiments.",
    )

    # Add the n_procs argument
    parser.add_argument(
        "-n",
        "--n_procs",
        type=int,
        required=True,
        help="The number of processor cores to use for running the experiments.",
    )

    parser.add_argument(
        "--no_sheets",
        action="store_true",
        help="Set this to prevent writing results to gsheets (i.e., they are only written to log files).",
    )

    # Parse the command-line arguments
    args = parser.parse_args()
    WRITE_TO_SHEET = not args.no_sheets
    # Call the function with the command-line arguments
    monitor_sheet_and_run_experiments(args.interval, args.n_procs)

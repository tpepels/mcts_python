from collections import Counter
import csv
import datetime
import glob
import json
import multiprocessing
import os
import re
import time

import gspread
from gspread_dataframe import set_with_dataframe

import numpy as np
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from prettytable import PrettyTable

from run_games import AIParams, run_multiple_game_experiments
from util import read_config

# Column indices in Google Sheets (1-based)
STATUS_COLUMN = 1
N_GAMES_COLUMN = 2
START_COLUMN = 3
COMPLETED_GAMES_COLUMN = 4
ERROR_GAMES_COLUMN = 5
WORKSHEET_COLUMN = 6

# Status values
PENDING = "Pending"
RUNNING = "Running"
COMPLETED = "Completed"
INTERRUPTED = "Interrupted"
RESUME = "Resume"


def mark_experiment_as_status(
    sheet: gspread.Worksheet,
    index,
    status,
    n_games=None,
    start_time=None,
    completed_games=0,
    error_games=0,
    worksheet_name=None,
):
    """
    Update the status of an experiment in the Google Sheets document. If the status is 'Running',
    it also updates the total number of games, the start time, the number of completed games, and
    the number of games with errors.

    Args:
        sheet (gspread.Worksheet): The Google Sheets document.
        index (int): The index of the experiment in the DataFrame (0-based).
        status (str): The new status of the experiment ('Pending', 'Running', or 'Completed').
        n_games (int, optional): The total number of games for the experiment. Set this parameter only once at the beginning of the experiment.
        start_time (datetime, optional): The start time of the experiment. If not provided and the status is 'Running',
                                         the current time is used.
        completed_games (int, optional): The number of completed games.
        error_games (int, optional): The number of games with errors.

    Raises:
        Exception: Any error occurred during the update operation.

    Returns:
        None
    """
    try:
        # We add 2 because Google Sheets has 1-based indexing and the first row is the header
        sheet.update_cell(index + 2, STATUS_COLUMN, status)
        if status in [RUNNING, COMPLETED]:
            if n_games:
                sheet.update_cell(index + 2, N_GAMES_COLUMN, n_games)  # Total number of games
            if start_time:
                sheet.update_cell(
                    index + 2, START_COLUMN, start_time.strftime("%Y-%m-%d %H:%M:%S")
                )  # Start time
            if completed_games > 0:
                sheet.update_cell(index + 2, COMPLETED_GAMES_COLUMN, completed_games)  # Completed games
            if error_games > 0:
                sheet.update_cell(index + 2, ERROR_GAMES_COLUMN, error_games)  # Games with errors
            if worksheet_name:
                sheet.update_cell(index + 2, WORKSHEET_COLUMN, worksheet_name)
    except Exception as e:
        print(f"Failed to mark experiment as {status}: {e}")


def parse_parameters(param_string: str, sheet: gspread.worksheet, index: int):
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
    try:
        params = json.loads(param_string)
        for key, value in params.items():
            if isinstance(value, str) and re.match(r"\[(.*)\]", value):  # Check if the value is a list
                params[key] = json.loads(value)  # Converts the string to list using json
            elif isinstance(value, str) and ":" in value:  # Check if the value is a range
                start, end, step = map(float, value.split(":"))
                params[key] = list(np.arange(start, end, step))
        return params

    except Exception as e:
        print(f"Failed to parse parameters: {e}")
        # Here, assuming that your ERROR_COLUMN is the 6th column. You can change this according to your sheet.
        ERROR_COLUMN = 6
        sheet.update_cell(index + 2, ERROR_COLUMN, f"Failed to parse parameters: {e}")
        return None


def expand_rows(df: pd.DataFrame):
    """
    Expand the rows of the DataFrame to represent each combination of parameter values (parameter set).

    Args:
        df (pd.DataFrame): A pandas DataFrame with parameter columns. If a column's first value is a list,
                           the entire column is treated as list-like.

    Returns:
        pd.DataFrame: A new pandas DataFrame with expanded rows.

    Example:
        input_df = pd.DataFrame({
            "param1": [[1, 2]],
            "param2": [[3, 4]]
        })
        expand_rows(input_df) would return:
        pd.DataFrame({
            "param1": [1, 2, 1, 2],
            "param2": [3, 3, 4, 4]
        })
    """
    param_columns = [col for col in df.columns if isinstance(df[col].iloc[0], list)]
    expanded_rows = []

    for _, row in df.iterrows():
        temp_df = pd.DataFrame()
        for col in param_columns:
            temp_df = pd.concat([temp_df, pd.DataFrame({col: row[col]})], axis=1)

        expanded_rows.append(temp_df)

    return pd.concat(expanded_rows, ignore_index=True)


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
    df = pd.DataFrame(data)

    # Save the original dataframe before applying the filter
    original_df = df.copy()

    # Only process experiments that are 'Pending'
    df = df[df["status"] == PENDING]

    # Parse the parameters
    df["game_params"] = df.apply(
        lambda row: parse_parameters(row["game_params"], sheet, row["original_index"]), axis=1
    )
    df["ai_params"] = df.apply(
        lambda row: parse_parameters(row["ai_params"], sheet, row["original_index"]), axis=1
    )

    # Filter out rows with parsing errors
    df = df[df["game_params"].notna() & df["ai_params"].notna()]

    # Expand rows for parameter combinations
    df_expanded = expand_rows(df)

    if not df_expanded.equals(df):
        # Write the expanded df back to google sheets so we keep track of individual experiments
        sheet.clear()
        set_with_dataframe(sheet, df_expanded)

    # Merge the filtered/expanded dataframe with the original dataframe
    result_df = pd.concat([original_df, df_expanded]).drop_duplicates()

    return result_df


def get_config_sheet(client: gspread.Client):
    """
    Fetch the Google Sheet specified in the configuration file.

    This function reads a config file to obtain the ID of the Google Sheet
    to be used for experiments, then uses the provided gspread client to open that sheet.

    Args:
        client (gspread.Client): A client for interacting with the Google Sheets API.

    Returns:
        gspread.models.Worksheet: The first worksheet of the Google Sheets document.
    """
    # Read the config file
    config = read_config()
    sheet_id = config.get("Sheets", "ExperimentListID")

    # Open the Google Sheets document
    sheet = client.open_by_key(sheet_id).sheet1

    return sheet


def is_experiment_running(current_experiment_process: multiprocessing.Process):
    """
    Check if there is a running experiment process.

    This function checks whether the given process is currently running.

    Args:
        current_experiment_process (multiprocessing.Process): The process to check.

    Returns:
        bool: True if the process is running, False otherwise.
    """
    return current_experiment_process and current_experiment_process.is_alive()


def run_new_experiment(row: pd.Series, index: int, sheet: gspread.Worksheet):
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
    # Create AIParams objects for each player
    p1_params = AIParams(ai_key=row["p1_ai_key"], ai_params=row["p1_ai_params"])
    p2_params = AIParams(ai_key=row["p2_ai_key"], ai_params=row["p2_ai_params"])

    game_params = row["game_params"]
    game_name = row["game_key"]

    # Create a new sheet for this experiment if it's a new one, else fetch the interrupted one
    if row["status"] == PENDING:
        if game_params and "board_size" in game_params:
            game_name_str = game_name + str(game_params["board_size"])
        else:
            game_name_str = game_name

        sheet_name = (
            f"{game_name_str}_{p1_params}_{p2_params}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        start_game = 0
    else:
        sheet_name = row["sheet_name"]
        start_game = row["completed_games"] + 1

    # Run the experiment in a separate process
    current_experiment_process = multiprocessing.Process(
        target=run_multiple_game_experiments,
        args=(
            row["n_games"],
            start_game,
            sheet_name,
            game_name,
            game_params,
            p1_params,
            p2_params,
            row["status"] != PENDING,
        ),
    )
    current_experiment_process.start()

    # Mark the new experiment as running
    mark_experiment_as_status(
        sheet,
        index,
        RUNNING,
        row["n_games"],
        datetime.datetime.now(),
        0,
        0,
        sheet_name,
    )

    return current_experiment_process


def update_running_experiment_status(row: pd.Series, index: int, sheet: gspread.Worksheet, is_running: bool):
    """
    Scan log files for status updates and update Google Sheet.

    This function scans the log files associated with a given experiment for status
    updates. It then updates the status of the experiment in the Google Sheets document
    based on the contents of the log files.

    Args:
        row (pandas.Series): A row from the DataFrame representing the experiments table.
        index (int): The index of the row in the DataFrame.
        sheet (gspread.Worksheet): The worksheet of the Google Sheets document.
        is_running (bool): True if the experiment is still running, false otherwise.
    """
    sheet_name = row["sheet_name"]
    completed_games = 0
    error_games = 0
    ai_stats = Counter()  # To hold cumulative statistics per AI
    log_files = glob.glob(f"log/games/{sheet_name}/?.log")

    # Open CSV file in append mode
    with open(f"log/games/{sheet_name}/_results.csv", "w", newline="") as f:
        writer = csv.writer(f)

        for log_file in log_files:
            with open(log_file, "r") as log_f:
                log_contents = log_f.readlines()
                game_number = log_file.split("/")[-1].split(".")[0]  # Get game number from log file name
                if (
                    "Experiment completed" in log_contents[-2]
                ):  # Assuming "Experiment completed" is second to last line
                    completed_games += 1
                    # Assuming the last line is "Game Over. Winner: Px [px_params]"
                    winner_info = log_contents[-1]  # Last line
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
    print_stats = PrettyTable(["AI", "Wins"])
    for ai, wins in ai_stats.items():
        print_stats.add_row([ai, wins])

    os.system("cls" if os.name == "nt" else "clear")
    print(sheet_name)
    print("*" * 10)
    print(print_stats)

    if is_running:
        mark_experiment_as_status(sheet, index, RUNNING, None, None, completed_games, error_games)
    else:
        mark_experiment_as_status(sheet, index, COMPLETED, None, None, completed_games, error_games)


def monitor_sheet_and_run_experiments(interval: int):
    """
    Continuously monitor a Google Sheets document for new experiments and execute them.

    This function will read experiments marked as 'Pending' or 'Interrupted' in the Google Sheets document,
    mark them as 'Running', run them in a separate process, and then mark them as 'Completed'. It checks
    the document every interval seconds for new experiments.

    If the function is interrupted (e.g., by a KeyboardInterrupt), it will attempt to terminate the currently
    running experiment process, mark the experiment as 'Interrupted' in the Google Sheets document, and stop monitoring.

    The Google Sheets document to monitor is specified in the application's configuration file.

    Raises:
        Exception: If any error occurs during the operation.

    Returns:
        None
    """
    # Use credentials to create a client to interact with the Google Drive API
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("client_secret.json", scopes)
    client = gspread.authorize(creds)

    # Get the Google Sheet from config
    sheet = get_config_sheet(client)

    # Running experiment process
    current_experiment_process = None

    try:
        while True:
            df = get_experiments_from_sheet(sheet)
            for index, row in df.iterrows():
                if row["status"] in [PENDING, RESUME] and not is_experiment_running(
                    current_experiment_process
                ):
                    current_experiment_process = run_new_experiment(row, index, sheet)
                elif row["Status"] == RUNNING:
                    update_running_experiment_status(
                        row, index, sheet, is_experiment_running(current_experiment_process)
                    )

                # Sleep for a while before checking for new experiments
                time.sleep(interval)
    except KeyboardInterrupt:
        print("Interrupted. Attempting to mark currently running experiment as interrupted.")
        if is_experiment_running(current_experiment_process):
            # Optionally: terminate the running experiment
            current_experiment_process.terminate()
            current_experiment_process.join(30)
            # Mark the running experiment as interrupted
            mark_experiment_as_status(sheet, index, INTERRUPTED)
        print("Stopped monitoring.")


if __name__ == "__main__":
    monitor_sheet_and_run_experiments(30)

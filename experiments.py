import datetime
import glob
import json
import multiprocessing
import re
import time

import gspread
import numpy as np
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

from run_games import AIParams, run_multiple_game_experiments
from util import read_config

# Column indices in Google Sheets (1-based)
STATUS_COLUMN = 1
N_GAMES_COLUMN = 2
START_COLUMN = 3
COMPLETED_GAMES_COLUMN = 4
ERROR_GAMES_COLUMN = 5

# Status values
PENDING = "Pending"
RUNNING = "Running"
COMPLETED = "Completed"
INTERRUPTED = "Interrupted"


def mark_experiment_as_status(
    sheet, index, status, n_games=None, start_time=None, completed_games=0, error_games=0
):
    """
    Update the status of an experiment in the Google Sheets document. If the status is 'Running',
    it also updates the total number of games, the start time, the number of completed games, and
    the number of games with errors.

    Args:
        sheet (gspread.models.Worksheet): The Google Sheets document.
        index (int): The index of the experiment in the DataFrame (0-based).
        status (str): The new status of the experiment ('Pending', 'Running', or 'Completed').
        n_games (int, optional): The total number of games for the experiment. Required if the status is 'Running'.
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
        if status == RUNNING:
            sheet.update_cell(index + 2, N_GAMES_COLUMN, n_games)  # Total number of games
            if start_time is None:
                start_time = datetime.datetime.now()
            sheet.update_cell(index + 2, START_COLUMN, start_time.strftime("%Y-%m-%d %H:%M:%S"))  # Start time
            sheet.update_cell(index + 2, COMPLETED_GAMES_COLUMN, completed_games)  # Completed games
            sheet.update_cell(index + 2, ERROR_GAMES_COLUMN, error_games)  # Games with errors
    except Exception as e:
        print(f"Failed to mark experiment as {status}: {e}")


def parse_parameters(param_string):
    """
    Parse a string containing parameter details into a dictionary. If a value is a JSON list or a range (in the format
    'start:end:step'), it's converted into a Python list.

    Args:
        param_string (str): A string containing parameter details. Expected to be a JSON-like string.

    Returns:
        dict: A dictionary containing the parsed parameters.
    """

    params = json.loads(param_string)
    for key, value in params.items():
        if isinstance(value, str) and re.match(r"\[(.*)\]", value):  # Check if the value is a list
            params[key] = json.loads(value)  # Converts the string to list using json
        elif isinstance(value, str) and ":" in value:  # Check if the value is a range
            start, end, step = map(float, value.split(":"))
            params[key] = list(np.arange(start, end, step))
    return params


def expand_rows(df):
    """
    Expand the rows of the DataFrame to represent each combination of parameter values.

    Args:
        df (pd.DataFrame): A pandas DataFrame with parameter columns. If a column's first value is a list,
                           the entire column is treated as list-like.

    Returns:
        pd.DataFrame: A new pandas DataFrame with expanded rows.
    """
    param_columns = [col for col in df.columns if isinstance(df[col].iloc[0], list)]
    expanded_rows = []

    for _, row in df.iterrows():
        temp_df = pd.DataFrame()
        for col in param_columns:
            temp_df = pd.concat([temp_df, pd.DataFrame({col: row[col]})], axis=1)

        expanded_rows.append(temp_df)

    return pd.concat(expanded_rows, ignore_index=True)


def get_experiments_from_sheet(sheet):
    """
    Read the experiment data from a Google Sheets document, filter out the experiments that are not 'Pending',
    parse their parameters, and expand the rows for parameter combinations.

    Args:
        sheet (gspread.models.Worksheet): The Google Sheets document.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the experiment data.
    """
    data = sheet.get_all_records()
    df = pd.DataFrame(data)

    # Only process experiments that are 'Pending'
    df = df[df["status"] == "Pending"]

    # Parse the parameters
    df["game_params"] = df["game_params"].apply(parse_parameters)
    df["ai_params"] = df["ai_params"].apply(parse_parameters)

    # Expand rows for parameter combinations
    df = expand_rows(df)

    return df


def monitor_sheet_and_run_experiments():
    """
    Continuously monitor a Google Sheets document for new experiments and run them. The Google Sheets document
    to monitor is read from the configuration file. It reads the experiments that are marked as 'Pending',
    marks them as 'Running', runs them, and then marks them as 'Completed'. After all 'Pending' experiments
    are processed, it waits for a minute before checking again. If interrupted during the wait, it stops monitoring.

    Raises:
        Exception: Any error occurred during the operation.

    Returns:
        None
    """
    # Use credentials to create a client to interact with the Google Drive API
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("client_secret.json", scopes)
    client = gspread.authorize(creds)

    # Read the config file
    config = read_config()
    sheet_id = config.get("Sheets", "ExperimentListID")

    # Open the Google Sheets document
    # The ID of a Google Sheets document is contained in the URL. For example, for a URL like this: https://docs.google.com/spreadsheets/d/1qPY_your_unique_id/edit#gid=0, the ID is 1qPY_your_unique_id.
    sheet = client.open_by_key(sheet_id).sheet1

    # Running experiment process
    current_experiment_process = None
    try:
        while True:
            df = get_experiments_from_sheet(sheet)
            for index, row in df.iterrows():
                # If experiment status is 'Pending', then run it, but if there is a running experiment, do not start a new one
                if (
                    row["Status"] == PENDING
                    and current_experiment_process
                    and not current_experiment_process.is_alive()
                ):
                    # Create AIParams objects for each player
                    p1_params = AIParams(ai_key=row["p1_ai_key"], ai_params=row["p1_ai_params"])
                    p2_params = AIParams(ai_key=row["p2_ai_key"], ai_params=row["p2_ai_params"])

                    game_params = row["game_params"]
                    game_name = row["game_key"]

                    # Create a new sheet for this experiment
                    if game_params and "board_size" in game_params:
                        game_name_str = game_name + str(game_params["board_size"])
                    else:
                        game_name_str = game_name

                    # TODO Write the sheet name to the status so we can use it later
                    sheet_name = f"{game_name_str}_{p1_params}_{p2_params}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

                    # Run the experiment in a separate process
                    current_experiment_process = multiprocessing.Process(
                        target=run_multiple_game_experiments,
                        args=(row["n_games"], sheet_name, game_name, game_params, p1_params, p2_params),
                    )
                    current_experiment_process.start()
                    # Mark the new experiment as running
                    mark_experiment_as_status(
                        sheet, index, RUNNING, row["n_games"], datetime.datetime.now().isoformat(), 0, 0
                    )

                elif row["Status"] == RUNNING:
                    # Scan log files for status messages
                    completed_games = 0
                    error_games = 0
                    log_files = glob.glob(f"log/games/{sheet_name}/?.log")
                    for log_file in log_files:
                        with open(log_file, "r") as f:
                            log_contents = f.read()
                            if "Experiment completed" in log_contents:
                                completed_games += 1
                            elif "Experiment error" in log_contents:
                                error_games += 1
                    if current_experiment_process and not current_experiment_process.is_alive():
                        # If the experiment has completed, mark it as completed
                        mark_experiment_as_status(
                            sheet, index, COMPLETED, row["n_games"], None, completed_games, error_games
                        )
                    else:
                        mark_experiment_as_status(
                            sheet, index, RUNNING, row["n_games"], None, completed_games, error_games
                        )

                # Sleep for a while before checking for new experiments
                time.sleep(30)
    except KeyboardInterrupt:
        print("Interrupted. Attempting to mark currently running experiment as interrupted.")
        if current_experiment_process and current_experiment_process.is_alive():
            # Optionally: terminate the running experiment
            current_experiment_process.terminate()
            current_experiment_process.join(30)
            # Mark the running experiment as interrupted
            mark_experiment_as_status(
                sheet, index, INTERRUPTED, row["n_games"], None, completed_games, error_games
            )
        print("Stopped monitoring.")


if __name__ == "__main__":
    monitor_sheet_and_run_experiments()

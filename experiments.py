import pandas as pd
import numpy as np
import gspread
import time
import json
import re

from oauth2client.service_account import ServiceAccountCredentials
from run_games import AIParams, run_multiple_game_experiments


import datetime

STATUS_COLUMN = 1
N_GAMES_COLUMN = 2
START_COLUMN = 3


def mark_experiment_as_running(sheet, index, n_games, start_time=None):
    """
    Mark an experiment as 'Running' in the Google Sheets document.

    Args:
        sheet: The Google Sheets document.
        index: The index of the experiment.
        n_games: The total number of games for the experiment.
        start_time: The start time of the experiment.
    """
    # We add 2 because Google Sheets has 1-based indexing and the first row is the header
    sheet.update_cell(index + 2, STATUS_COLUMN, "Running")
    sheet.update_cell(index + 2, N_GAMES_COLUMN, n_games)  # Total number of games

    if start_time is None:
        start_time = datetime.datetime.now()
    sheet.update_cell(index + 2, START_COLUMN, start_time.strftime("%Y-%m-%d %H:%M:%S"))  # Start time


def mark_experiment_as_completed(sheet, index):
    """
    Mark an experiment as 'Completed' in the Google Sheets document.

    Args:
        sheet: The Google Sheets document.
        index: The index of the experiment.
    """
    # We add 2 because Google Sheets has 1-based indexing and the first row is the header
    sheet.update_cell(index + 2, STATUS_COLUMN, "Completed")


def parse_parameters(param_string):
    """
    Parse a string containing parameter details into a dictionary.

    Args:
        param_string: String containing parameter details. This could be a JSON-like string.

    Returns:
        A dictionary containing the parsed parameters.
    """
    params = json.loads(param_string)

    for key, value in params.items():
        if isinstance(value, str) and re.match(r"\[(.*)\]", value):  # Check if the value is a list
            params[key] = eval(value)  # Converts the string to list
        elif isinstance(value, str) and ":" in value:  # Check if the value is a range
            start, end, step = map(float, value.split(":"))
            params[key] = list(np.arange(start, end, step))

    return params


def expand_rows(df):
    """
    Expand the rows of the DataFrame to represent each combination of parameter values.

    Args:
        df: pandas DataFrame with parameter columns.

    Returns:
        A new pandas DataFrame with expanded rows.
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
    Read the experiment data from a Google Sheets document.

    Args:
        sheet: The Google Sheets document.
    Returns:
        A pandas DataFrame containing the experiment data.
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
    Continuously monitor a Google Sheets document for new experiments and run them.
    """
    # Use credentials to create a client to interact with the Google Drive API
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("client_secret.json", scopes)
    client = gspread.authorize(creds)

    # Open the Google Sheets document
    sheet = client.open("Experiment List").sheet1

    while True:
        df = get_experiments_from_sheet(sheet)

        for index, row in df.iterrows():
            # If experiment status is 'Pending', then run it
            if row["Status"] == "Pending":
                # Mark the experiment as 'Running'
                mark_experiment_as_running(sheet, index, row["n_games"])

                # Create AIParams objects for each player
                p1_params = AIParams(ai_key=row["p1_ai_key"], ai_params=row["p1_ai_params"])
                p2_params = AIParams(ai_key=row["p2_ai_key"], ai_params=row["p2_ai_params"])

                # Run the experiment
                run_multiple_game_experiments(
                    n_games=row["n_games"],
                    game_key=row["game_key"],
                    game_params=row["game_params"],
                    p1_params=p1_params,
                    p2_params=p2_params,
                )
                # Mark the experiment as 'Completed'
                mark_experiment_as_completed(sheet, index)

        # Sleep for a while before checking for new experiments
        time.sleep(60)


if __name__ == "__main__":
    monitor_sheet_and_run_experiments()

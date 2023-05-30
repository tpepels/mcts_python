from functools import partial
import time

from ai.alpha_beta import AlphaBetaPlayer
from ai.mcts import MCTSPlayer
from games.tictactoe import TicTacToeGameState, evaluate_tictactoe
from games.breakthrough import (
    BreakthroughGameState,
    evaluate_breakthrough,
    lorenz_evaluation,
    lorenz_enhanced_evaluation,
)
from games.amazons import AmazonsGameState, evaluate_amazons, evaluate_amazons_lieberum
from games.kalah import KalahGameState, evaluate_kalah, evaluate_kalah_enhanced

# Contains all possible evaluation functions for use in factory
eval_dict = {
    evaluate_tictactoe.__name__: evaluate_tictactoe,
    evaluate_breakthrough.__name__: evaluate_breakthrough,
    lorenz_evaluation.__name__: lorenz_evaluation,
    lorenz_enhanced_evaluation.__name__: lorenz_enhanced_evaluation,
    evaluate_amazons.__name__: evaluate_amazons,
    evaluate_amazons_lieberum.__name__: evaluate_amazons_lieberum,
    evaluate_kalah.__name__: evaluate_kalah,
    evaluate_kalah_enhanced.__name__: evaluate_kalah_enhanced,
}

# Contains all possible games for use in factory
game_dict = {
    "tictactoe": TicTacToeGameState,
    "breakthrough": BreakthroughGameState,
    "amazons": AmazonsGameState,
    "kalah": KalahGameState,
}

# Contains all possible ai players for use in factory
player_dict = {"alphabeta": AlphaBetaPlayer, "mcts": MCTSPlayer}


def run_game(
    game_key,
    ai_key,
    eval_keys,
    game_params,
    ai1_params,
    ai2_params,
    eval_params_p1: dict = None,
    eval_params_p2: dict = None,
):
    """
    Run a game simulation between two AI players using provided parameters.

    This function initializes a game state and two AI players, then conducts a turn-based game until
    a terminal state is reached. After each action, it prints the player, chosen action, evaluation
    value, and the time taken for the AI to decide on the action.

    At the end of the game, it prints the final game state and the game's result for player 1.

    Parameters
    ----------
    game_key : str
        The key to select the game from the `game_dict`. Should correspond to one of the supported game types:
        "tictactoe", "breakthrough", "amazons", or "kalah".

    ai_key : str
        The key to select the AI player class from the `player_dict`. Should be either "alphabeta" or "mcts".

    eval_keys : dict
        A dictionary containing two keys "p1" and "p2", with values corresponding to the keys of the `eval_dict`.
        These are used to select the evaluation function for each player.

    game_params : dict
        A dictionary of parameters to pass to the game state initializer. The required parameters depend on the
        specific game.

    ai_params : dict
        A dictionary of parameters to pass to the AI player initializers. The required parameters depend on the
        specific AI class. Common parameters might include "depth" for the search depth in an Alpha-Beta player,
        or "num_simulations" for the number of simulations to run in an MCTS player.

    eval_params_p1, eval_params_p2: dict
        Two dictionaries that can be used to pass parameters to the respective evaluation functions (m and a).
    """
    # Fetch the corresponding classes from the dictionaries
    game_class = game_dict[game_key]
    ai_class = player_dict[ai_key]

    # Fetch the corresponding evaluation functions
    eval_function_p1 = eval_dict[eval_keys["p1"]]
    eval_function_p2 = eval_dict[eval_keys["p2"]]

    if eval_params_p1 is not None:
        eval_function_p1 = partial(eval_function_p1, **eval_params_p1)
    if eval_params_p2 is not None:
        eval_function_p2 = partial(eval_function_p2, **eval_params_p2)

    # Initialize game state
    game = game_class(**game_params)

    # Initialize two AI players
    p1 = ai_class(player=1, evaluate=eval_function_p1, **ai1_params)
    p2 = ai_class(player=2, evaluate=eval_function_p2, **ai2_params)

    while not game.is_terminal():
        start_time = time.time()
        if game.player == 1:
            action, v = p1.best_action(game)
            elapsed_time = time.time() - start_time
            print(
                f"Player 1's turn. Chosen action: {action} v:{v:.3f}. Search took {elapsed_time:.1f} seconds."
            )
        else:
            action, v = p2.best_action(game)
            elapsed_time = time.time() - start_time
            print(
                f"Player 2's turn. Chosen action: {action} v:{v:3f}. Search took {elapsed_time:.1f} seconds."
            )

        game = game.apply_action(action)

        print(game.visualize())

    result = game.get_reward(1)
    print(f"Game Over. Result for p1: {result}")


from typing import List, Tuple
import time


def run_game_experiment(
    game_key: str,
    ai_key: str,
    eval_keys: dict,
    game_params: dict,
    ai_params: dict,
    eval_params_p1: dict = None,
    eval_params_p2: dict = None,
) -> Tuple[str, str, float, Tuple[float, float], int]:
    # Fetch the corresponding classes from the dictionaries
    game_class = game_dict[game_key]
    ai_class = player_dict[ai_key]

    # Fetch the corresponding evaluation functions
    eval_function_p1 = eval_dict[eval_keys["p1"]]
    eval_function_p2 = eval_dict[eval_keys["p2"]]

    if eval_params_p1 is not None:
        eval_function_p1 = partial(eval_function_p1, **eval_params_p1)
    if eval_params_p2 is not None:
        eval_function_p2 = partial(eval_function_p2, **eval_params_p2)

    # Initialize game state
    game = game_class(**game_params)

    # Initialize two AI players
    p1 = ai_class(player=1, evaluate=eval_function_p1, **ai_params)
    p2 = ai_class(player=2, evaluate=eval_function_p2, **ai_params)

    # Initialize stats
    setup = f"Game: {game_key} with parameters {game_params}. AI: {ai_key} with parameters {ai_params} and evaluation functions {eval_keys}"
    if eval_params_p1 is not None:
        setup += f" with p1 parameters {eval_params_p1}"
    if eval_params_p2 is not None:
        setup += f" with p2 parameters {eval_params_p2}"

    game_output = []
    times = {1: [], 2: []}
    start_time_total = time.time()

    while not game.is_terminal():
        start_time_move = time.time()
        if game.player == 1:
            action, v = p1.best_action(game)
        else:
            action, v = p2.best_action(game)

        elapsed_time_move = time.time() - start_time_move
        times[game.player].append(elapsed_time_move)

        game_output.append(
            f"Player {game.player}, action: {action}, v: {v:.3f}, time: {elapsed_time_move:.3f}"
        )
        game = game.apply_action(action)
        game_output.append(f"State: {eval_function_p1(game, 1)=} {eval_function_p2(game, 2)=}")
        game_output.append(game.visualize())

    total_time = time.time() - start_time_total
    avg_time_per_move = (sum(times[1]) / len(times[1]), sum(times[2]) / len(times[2]))
    result = game.get_reward(1)

    return setup, "\n".join(game_output), total_time, avg_time_per_move, result


import gspread
import multiprocessing
from google.oauth2.service_account import Credentials


def run_multiple_game_experiments(
    n_games: int, game_key: str, ai_key: str, eval_keys: dict, game_params: dict, ai_params: dict
):
    # Use credentials to create a client to interact with the Google Drive API
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file("client_secret.json", scopes=scopes)
    client = gspread.authorize(creds)

    # Create a new Google Sheets document
    title = f"Game Results for {n_games} games with {game_key} and {ai_key}"
    sheet = client.create(title)
    sheet.share("tpepels@gmail.com", perm_type="user", role="writer")  # Share the sheet with your account

    # Write headers
    worksheet = sheet.get_worksheet(0)
    worksheet.insert_row(
        ["Experiment", "Setup", "Game Output", "Total Time", "Avg Time p1", "Avg Time p2", "Result"], 1
    )

    def run_single_game(i, game_key, ai_key, eval_keys, game_params, ai_params):
        setup, game_output, total_time, avg_time_per_move, result = run_game_experiment(
            game_key, ai_key, eval_keys, game_params, ai_params
        )
        avg_time_p1, avg_time_p2 = avg_time_per_move
        # Append the result to the sheet
        worksheet.append_row([i, setup, game_output, total_time, avg_time_p1, avg_time_p2, result])

    # Run n_games instances of the game in parallel, distributing across the available CPU cores
    with multiprocessing.Pool() as pool:
        pool.starmap(
            run_single_game,
            [(i, game_key, ai_key, eval_keys, game_params, ai_params) for i in range(n_games)],
        )

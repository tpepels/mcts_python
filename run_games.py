import datetime
import multiprocessing
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple, Type

import gspread
from google.oauth2.service_account import Credentials

from ai.ai_player import AIPlayer
from ai.alpha_beta import AlphaBetaPlayer
from ai.mcts import MCTSPlayer
from games.amazons import AmazonsGameState, evaluate_amazons, evaluate_amazons_lieberum
from games.breakthrough import (
    BreakthroughGameState,
    evaluate_breakthrough,
    lorenz_enhanced_evaluation,
    lorenz_evaluation,
)
from games.gamestate import GameState, draw, loss, win
from games.kalah import KalahGameState, evaluate_kalah, evaluate_kalah_enhanced
from games.tictactoe import TicTacToeGameState, evaluate_tictactoe
from util import log_exception_handler, read_config, redirect_print_to_log

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


@dataclass
class AIParams:
    """Class for holding AI parameters.

    Attributes:
        ai_key (str): The key for the AI algorithm.
        eval_key (str): The key for the evaluation function.
        ai_params (Dict[str, Any]): The parameters for the AI algorithm.
        eval_params (Dict[str, Any]): The parameters for the evaluation function.
    """

    ai_key: str
    eval_key: str
    ai_params: Optional[Dict[str, Any]] = None
    eval_params: Optional[Dict[str, Any]] = None

    def __str__(self):
        """Generate string representation of AI parameters."""
        string_repr = f"AI: {self.ai_key}"
        if self.ai_params is not None:
            string_repr += f" with parameters {self.ai_params}"

        string_repr += f" and evaluation function {self.eval_key}"

        if self.eval_params is not None:
            string_repr += f" with parameters {self.eval_params}."
        return string_repr


def init_game_and_players(
    game_key: str, game_params: Optional[Dict[str, Any]], p1_params: AIParams, p2_params: AIParams
) -> Tuple[Type[GameState], AIPlayer, AIPlayer]:
    """Initialize game and players based on given parameters.

    Args:
        game_key (str): The key for the game.
        game_params (Dict[str, Any]): The parameters for the game.
        p1_params (AIParams): The parameters for player 1's AI.
        p2_params (AIParams): The parameters for player 2's AI.

    Returns:
        Tuple[Type[GameState], AIPlayer, AIPlayer]: The initialized game and players.
    """
    game_class = game_dict[game_key]
    game: GameState = game_class(**game_params)

    ai_class = player_dict[p1_params.ai_key]
    eval_function_p1 = eval_dict[p1_params.eval_key]
    if p1_params.eval_params is not None:
        eval_function_p1 = partial(eval_function_p1, **p1_params.eval_params)
    p1: AIPlayer = ai_class(player=1, evaluate=eval_function_p1, **p1_params.ai_params)

    ai_class = player_dict[p2_params.ai_key]
    eval_function_p2 = eval_dict[p2_params.eval_key]
    if p2_params.eval_params is not None:
        eval_function_p2 = partial(eval_function_p2, **p2_params.eval_params)
    p2: AIPlayer = ai_class(player=2, evaluate=eval_function_p2, **p2_params.ai_params)

    return game, p1, p2


def play_game_until_terminal(game, player1, player2, callback=None):
    """
    Play the game with the provided players and return the result.

    Args:
        game (GameState): The game to be played.
        player1 (AIPlayer): The first player.
        player2 (AIPlayer): The second player.

    Returns:
        int: The result of the game. gamestate.draw for a draw, gamestate.win if player 1 won, and gamestate.loss if player 2 won.
    """
    current_player: AIPlayer = player1
    while not game.is_terminal():
        # Get the best action for the current player
        action, _ = current_player.best_action(game)

        # Apply the action to get the new game state
        game = game.apply_action(action)

        # Call the callback function if any
        if callback is not None:
            callback(current_player, action, game)

        # Switch the current player
        current_player = player2 if current_player == player1 else player1

    return game.get_reward(1)


def run_game(game_key: str, game_params: Dict[str, Any], p1_params: AIParams, p2_params: AIParams) -> None:
    """Run the game with two AI players.

    Args:
        game_key (str): The key for the game.
        game_params (Dict[str, Any]): The parameters for the game.
        p1_params (AIParams): The parameters for player 1's AI.
        p2_params (AIParams): The parameters for player 2's AI.
    """

    def callback(player, action, game):
        print(f"Player {player} chosen action: {action}. Current game state: \n {game.visualize()}")
        if game.get_reward(1) == win:
            print("Game Over. Winner: P1")
        elif game.get_reward(1) == loss:
            print("Game Over. Winner: P2")
        else:
            print("Game Over. Draw")

    game, p1, p2 = init_game_and_players(game_key, game_params, p1_params, p2_params)
    return play_game_until_terminal(game, p1, p2, callback=callback)


@log_exception_handler
def run_game_experiment(game_key: str, game_params: Dict[str, Any], p1_params: AIParams, p2_params: AIParams):
    """Run a game experiment with two AI players, and return detailed game information.

    The function initializes the game and players based on the provided parameters. It then runs the game, recording
    the moves made by the AI players and tracking the time taken for each move. The function records the game state
    after each move. The game continues until it reaches a terminal state.

    Args:
        game_key (str): The key identifying the type of game to be played.
        game_params (Dict[str, Any]): The parameters for initializing the game.
        p1_params (AIParams): The parameters for initializing the AI of player 1.
        p2_params (AIParams): The parameters for initializing the AI of player 2.

    Returns:
        setup (str): Description of the game setup.
        game_output (str): A string with line-by-line visualization of the game state after each move.
        total_time (float): The total time the game took in seconds.
        avg_time_per_move (Tuple[float, float]): A tuple containing the average time per move for player 1 and player 2, respectively.
        n_moves (int): The total number of moves made in the game.
        result (int): The result of the game. 1 if player 1 won, 2 if player 2 won, and 0 if the game was a draw.
    """

    game, p1, p2 = init_game_and_players(game_key, game_params, p1_params, p2_params)
    # Initialize stats
    setup = f"Game: {game_key} with parameters {game_params}. Player 1: {p1_params}. Player 2: {p2_params}"

    times = {1: [], 2: []}
    n_moves = 0

    def callback(current_player, action, game):
        nonlocal n_moves
        start_time_move = time.time()
        v = current_player.evaluate(game)
        elapsed_time_move = time.time() - start_time_move
        times[game.player].append(elapsed_time_move)

        print(f"Player {game.player}, action: {action}, v: {v:.1f}, time: {elapsed_time_move:.1f}")
        print(game.visualize())
        n_moves += 1

    start_time_total = time.time()
    result = play_game_until_terminal(game, p1, p2, callback=callback)
    total_time = time.time() - start_time_total

    avg_time_per_move = (sum(times[1]) / len(times[1]), sum(times[2]) / len(times[2]))

    if result == win:
        result = 1
        print("Game Over. Winner: P1")
    elif result == loss:
        result = 2
        print("Game Over. Winner: P2")
    else:
        result = 0
        print("Game Over. Draw")

    return setup, total_time, avg_time_per_move, n_moves, result


def get_authenticated_sheets_client() -> gspread.client.Client:
    """Authorize a client to interact with the Google Drive API."""
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file("client_secret.json", scopes=scopes)
    return gspread.authorize(creds)


def run_multiple_game_experiments(
    n_games: int,
    worksheet_name: str,
    game_key: str,
    p1_params: AIParams,
    p2_params: AIParams,
    game_params: dict[str, Any],
) -> None:
    """Run multiple game experiments and log the results in a Google Sheets document.

    Args:
        n_games (int): The number of games to run.
        game_key (str): The key for the game.
        p1_params (AIParams): The parameters for player 1's AI.
        p2_params (AIParams): The parameters for player 2's AI.
        game_params (dict[str, Any]): The parameters for the game.
    """
    client = get_authenticated_sheets_client()

    # Create a unique Google Sheets document
    main_sheet = client.create(worksheet_name)
    config = read_config()
    main_sheet.share(config["Share"]["GoogleAccount"], perm_type="user", role="writer")

    # Write headers
    worksheet = main_sheet.get_worksheet(0)
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
        ],
        1,
    )

    # Write formulas to keep track of the results per AI (not per seat)
    worksheet.update_acell("K1", "Winrate (per AI)")
    worksheet.update_acell("K2", '=AVERAGEIF(I2:I, "1")')
    worksheet.update_acell("L1", "95% CI")
    worksheet.update_acell("L2", "=CONFIDENCE.T(0.05, STDEV(I2:I), COUNTA(I2:I))")
    worksheet.update_acell("M1", "Average Time per Move")
    worksheet.update_acell("M2", "=AVERAGE(E2:E, F2:F)")
    worksheet.update_acell("N1", "Average # of moves")
    worksheet.update_acell("N2", "=AVERAGE(D2:D)")

    def run_single_game(
        i: int,
        game_key: str,
        game_params: dict[str, Any],
        p1_params: AIParams,
        p2_params: AIParams,
        players_switched: bool,
    ) -> None:
        """Run a single game experiment and log the results in the worksheet.

        Args:
            i (int): The game number.
            game_key (str): The key for the game.
            game_params (dict[str, Any]): The parameters for the game.
            p1_params (AIParams): The parameters for player 1's AI.
            p2_params (AIParams): The parameters for player 2's AI.
            work_sheet_name (str): The name of the sheet to place the results in.
            players_switched (bool): Whether player 1 and 2 are switched.
        """
        try:
            with redirect_print_to_log(f"log/games/{worksheet_name}/{i}.log") as log_file:
                setup, total_time, avg_time_per_move, n_moves, result = run_game_experiment(
                    game_key, game_params, p1_params, p2_params
                )
                # Write a status message to the log file
                log_file.write("Experiment completed\n")
        except Exception as e:
            with open(f"log/games/{worksheet_name}/{i}.log", "a") as log_file:
                log_file.write(f"Experiment error: {e}\n")

        avg_time_p1, avg_time_p2 = avg_time_per_move
        client = get_authenticated_sheets_client()

        # Retrieve the pre-created Google Sheets document
        game_sheet = client.open(worksheet_name)

        # Write the game result to the sheet
        worksheet = game_sheet.get_worksheet(0)

        # Append the result to the sheet (view from p1)
        # Keep track of results per AI not for p1/p2
        p1_result = (3 - result) if players_switched else result
        p2_result = result if players_switched else (3 - result)

        try:
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
                ]
            )
        except Exception as e:
            print(f"An error occurred while writing to the sheet: {e}")

    # The boolean parameter keeps track of whether the seats were switched
    games_params = [
        (i, game_key, game_params, p1_params, p2_params, False)
        if i < n_games / 2
        else (i, game_key, game_params, p2_params, p1_params, True)
        for i in range(n_games)
    ]

    # Shuffle the game parameters to randomize the order (so intermediate results can be interpreted better)
    random.shuffle(games_params)
    games_params = [(i, *params) for i, params in enumerate(games_params)]

    # Run n_games instances of the game in parallel, distributing across the available CPU cores
    with multiprocessing.Pool() as pool:
        pool.starmap(run_single_game, games_params)  # TODO Is pool.starmap a blocking call?

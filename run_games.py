import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple, Type

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
from games.gamestate import GameState
from games.kalah import KalahGameState, evaluate_kalah, evaluate_kalah_enhanced
from games.tictactoe import TicTacToeGameState, evaluate_tictactoe
from util import log_exception_handler

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


def run_game(game_key: str, game_params: Dict[str, Any], p1_params: AIParams, p2_params: AIParams) -> None:
    """Run the game with two AI players.

    Args:
        game_key (str): The key for the game.
        game_params (Dict[str, Any]): The parameters for the game.
        p1_params (AIParams): The parameters for player 1's AI.
        p2_params (AIParams): The parameters for player 2's AI.
    """
    game, p1, p2 = init_game_and_players(game_key, game_params, p1_params, p2_params)

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


@log_exception_handler
def run_game_experiment(game_key: str, game_params: Dict[str, Any], p1_params: AIParams, p2_params: AIParams):
    """Run a game experiment with two AI players.

    Args:
        game_key (str): The key for the game.
        game_params (Dict[str, Any]): The parameters for the game.
        p1_params (AIParams): The parameters for player 1's AI.
        p2_params (AIParams): The parameters for player 2's AI.
    """
    game, p1, p2 = init_game_and_players(game_key, game_params, p1_params, p2_params)

    # Initialize stats
    setup = f"Game: {game_key} with parameters {game_params}. Player 1: {p1_params}. Player 2: {p2_params}"

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
            f"Player {game.player}, action: {action}, v: {v:.1f}, time: {elapsed_time_move:.1f}"
        )
        game = game.apply_action(action)
        game_output.append(game.visualize())

    total_time = time.time() - start_time_total
    avg_time_per_move = (sum(times[1]) / len(times[1]), sum(times[2]) / len(times[2]))
    result = game.get_reward(1)

    return setup, "\n".join(game_output), total_time, avg_time_per_move, result


import multiprocessing

import gspread
from google.oauth2.service_account import Credentials


def get_authenticated_sheets_client() -> gspread.client.Client:
    """Authorize a client to interact with the Google Drive API."""
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file("client_secret.json", scopes=scopes)
    return gspread.authorize(creds)


def run_multiple_game_experiments(
    n_games: int,
    game_key: str,
    p1_params: AIParams,
    p2_params: AIParams,
    game_params: Dict[str, Any],
) -> None:
    """Run multiple game experiments and log the results in a Google Sheets document.

    Args:
        n_games (int): The number of games to run.
        game_key (str): The key for the game.
        p1_params (AIParams): The parameters for player 1's AI.
        p2_params (AIParams): The parameters for player 2's AI.
        game_params (Dict[str, Any]): The parameters for the game.
    """
    client = get_authenticated_sheets_client()
    # Create a new Google Sheets document
    title = f"Game Results for {n_games} games with {game_key}"
    sheet = client.create(title)
    sheet.share("tpepels@gmail.com", perm_type="user", role="writer")  # Share the sheet with your account

    # TODO Hier was je gebleven. Er moeten nog cellen in de sheets die stats bijhouden.
    # TODO Maar omdat de seats geswitched worden moet je kijken hoe je dit kan oplossen
    # TODO Misschien met een functie, of de resultaten altijd in view van 1 van de twee spelers houden of zoiets

    # Write headers
    worksheet = sheet.get_worksheet(0)
    worksheet.insert_row(
        [
            "Experiment",
            "Setup",
            "Game Output",
            "Total Time",
            "Avg Time p1",
            "Avg Time p2",
            "Result",
            "Player 1",
            "Player 2",
        ],
        1,
    )

    def run_single_game(
        i: int,
        game_key: str,
        game_params: Dict[str, Any],
        p1_params: AIParams,
        p2_params: AIParams,
        work_sheet_name: str,
    ) -> None:
        """Run a single game experiment and log the results in the worksheet.

        Args:
            i (int): The game number.
            game_key (str): The key for the game.
            game_params (Dict[str, Any]): The parameters for the game.
            p1_params (AIParams): The parameters for player 1's AI.
            p2_params (AIParams): The parameters for player 2's AI.
        """
        setup, game_output, total_time, avg_time_per_move, result = run_game_experiment(
            game_key, game_params, p1_params, p2_params
        )
        avg_time_p1, avg_time_p2 = avg_time_per_move

        client = get_authenticated_sheets_client()
        # Retrieve the pre-created Google Sheets document
        sheet = client.open(work_sheet_name)
        # Write the game result to the sheet
        worksheet = sheet.get_worksheet(0)
        # Append the result to the sheet
        worksheet.append_row(
            [
                i,
                setup,
                game_output,
                total_time,
                avg_time_p1,
                avg_time_p2,
                result,
                str(p1_params),
                str(p2_params),
            ]
        )

    games_params = [
        (game_key, game_params, p1_params, p2_params, title)
        if i < n_games / 2
        else (game_key, game_params, p2_params, p1_params, title)
        for i in range(n_games)
    ]
    # Shuffle the game parameters to randomize the order
    random.shuffle(games_params)
    games_params = [(i, *params) for i, params in enumerate(games_params)]
    # Run n_games instances of the game in parallel, distributing across the available CPU cores
    with multiprocessing.Pool() as pool:
        pool.starmap(run_single_game, games_params)

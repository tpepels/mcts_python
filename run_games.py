# cython: language_level=3

from array import array
import gc
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ai.alpha_beta import AlphaBetaPlayer
from ai.mcts import MCTSPlayer
from default_params import DEFAULT_SETTINGS

from games.amazons import AmazonsGameState
from games.blokus import BlokusGameState
from games.breakthrough import BreakthroughGameState

from cython.cimports.includes import GameState, loss, win

from games.kalah import KalahGameState
from games.tictactoe import TicTacToeGameState
from util import log_exception_handler


class AIPlayer:
    """Base class for AI players."""

    def best_action(self, state):
        pass

    def print_cumulative_statistics(self):
        pass


# Contains all possible games for use in factory
game_dict = {
    "tictactoe": TicTacToeGameState,
    "ninarow": TicTacToeGameState,
    "breakthrough": BreakthroughGameState,
    "amazons": AmazonsGameState,
    "kalah": KalahGameState,
    "blokus": BlokusGameState,
}

# Contains all possible ai players for use in factory
player_dict = {"alphabeta": AlphaBetaPlayer, "mcts": MCTSPlayer}


@dataclass
class AIParams:
    """Class for holding AI parameters.

    Attributes:
        ai_key (str): The key for the AI algorithm.
        max_player: (int): The ai's player.
        ai_params (Dict[str, Any]): The parameters for the AI algorithm.
        eval_params (Dict[str, Any]): The parameters for the evaluation function.
    """

    ai_key: str
    max_player: int
    game_name: str
    eval_params: Dict[str, Any]
    ai_params: Optional[Dict[str, Any]]
    transposition_table_size: int = 2**16

    def __post_init__(self):
        game_algorithm_combo = (self.game_name, self.ai_key)
        if game_algorithm_combo in DEFAULT_SETTINGS:
            defaults = DEFAULT_SETTINGS[game_algorithm_combo]

            if self.ai_params.get("no_defaults", False):
                print(f"Using no defaults for player {self.max_player}/{self.ai_key}")
                self.ai_params.pop("no_defaults")
            else:
                # Apply defaults for ai_params
                for key, default_value in defaults.get("ai_params", {}).items():
                    if key not in self.ai_params:
                        print(
                            f"Using default ai value {default_value} for {key} for player {self.max_player}/{self.ai_key}"
                        )
                        self.ai_params[key] = default_value
            if self.eval_params is not None and self.eval_params.get("no_defaults", False):
                print(f"Using no defaults for player {self.max_player}/{self.ai_key}")
                self.eval_params.pop("no_defaults")
            else:
                # Apply defaults for eval_params
                for key, default_value in defaults.get("eval_params", {}).items():
                    if key not in self.eval_params:
                        print(
                            f"Using default eval value {default_value} for {key} for player {self.max_player}/{self.ai_key}"
                        )
                        self.eval_params[key] = default_value

    def __str__(self):
        """Generate string representation of AI parameters."""
        string_repr = f"{self.ai_key}"
        if self.ai_params:
            string_repr += f" {d_to_s(self.ai_params)}"
        if self.eval_params:
            string_repr += f" {d_to_s(self.eval_params)}"
        return string_repr


def d_to_s(d):
    return ", ".join([str(k) + ":" + str(v) for k, v in d.items()])


def init_game_and_players(
    game_key: str, game_params: Optional[Dict[str, Any]], p1_params: AIParams, p2_params: AIParams
) -> Tuple[GameState, AIPlayer, AIPlayer]:
    """Initialize game and players based on given parameters.

    Args:
        game_key (str): The key for the game.
        game_params (Dict[str, Any]): The parameters for the game.
        p1_params (AIParams): The parameters for player 1's AI.
        p2_params (AIParams): The parameters for player 2's AI.

    Returns:
        Tuple[Type[GameState], AIPlayer, AIPlayer]: The initialized game and players.
    """
    game = init_game(game_key, game_params)
    p1_params.transposition_table_size = game.transposition_table_size
    p2_params.transposition_table_size = game.transposition_table_size
    p1 = init_ai_player(p1_params, game.param_order, game.default_params)
    p2 = init_ai_player(p2_params, game.param_order, game.default_params)
    return game, p1, p2


def init_game(game_key: str, game_params: Optional[Dict[str, Any]]) -> GameState:
    """Initialize the game based on given parameters.

    Args:
        game_key (str): The key for the game.
        game_params (Dict[str, Any]): The parameters for the game.

    Returns:
        Type[GameState]: The initialized game.
    """
    game_class = game_dict[game_key]
    game: GameState = game_class(**game_params)
    return game


def init_ai_player(
    params: AIParams,
    eval_param_order: dict[str, float],
    default_params,
) -> AIPlayer:
    """Initialize an AI player based on given parameters.

    Args:
        params (AIParams): The parameters for the AI player.
        player (int): The player number (1 or 2 for a 2-player game)
        transposition_table_size (int): The size of the transposition table.

    Returns:
        AIPlayer: The initialized AI player.
    """
    ai_class = player_dict[params.ai_key]

    assert params.ai_params is not None, f"AI parameters must be provided {params=}"

    # Create a python double array containing the evaluation function parameters in the order given in eval_param_order
    eval_params = array("d", [0.0] * len(eval_param_order))  # Initialize with zeros
    for param_name, index in eval_param_order.items():
        eval_params[index] = params.eval_params.get(param_name, default_params[index])

    player: AIPlayer = ai_class(
        player=params.max_player,
        eval_params=eval_params,
        transposition_table_size=params.transposition_table_size,
        **params.ai_params,
    )
    return player


def play_game_until_terminal(game: GameState, player1: AIPlayer, player2: AIPlayer, callback=None):
    """
    Play the game with the provided players and return the result.

    Args:
        game (GameState): The game to be played.
        player1 (AIPlayer): The first player.
        player2 (AIPlayer): The second player.

    Returns:
        int: The result of the game. gamestate.draw for a draw, gamestate.win if player 1 won, and gamestate.loss if player 2 won.
    """
    # Use os.urandom() to generate a cryptographically secure random seed
    seed_bytes = os.urandom(8)  # Generate 8 random bytes
    seed = int.from_bytes(seed_bytes, "big")  # Convert bytes to an integer

    # Set the random seed
    random.seed(seed)
    print(f"Random seed set to: {seed}")

    current_player: AIPlayer = player1
    turns = 1
    while not game.is_terminal():
        # Get the best action for the current player
        action, _ = current_player.best_action(game)

        assert action is not None, f"Player {current_player} returned None as best action{turns=}"
        assert action != (), f"Player {current_player} returned () as best action {turns=}"

        # Apply the action to get the new game state
        game = game.apply_action(action)
        gc.collect()
        # Call the callback function if any
        if callback is not None:
            callback(current_player, action, game, time.time())

        # Switch the current player
        current_player = player2 if game.player == 2 else player1
        turns += 1

    return game.get_reward(1)


def run_game(
    game_key: str,
    game_params: Dict[str, Any],
    p1_params: AIParams,
    p2_params: AIParams,
    pause=False,
    debug=False,
) -> float:
    """Run the game with two AI players.

    Args:
        game_key (str): The key for the game.
        game_params (Dict[str, Any]): The parameters for the game.
        p1_params (AIParams): The parameters for player 1's AI.
        p2_params (AIParams): The parameters for player 2's AI.
    """
    max_eval = -float("inf")
    n_moves = 0

    def callback(player, action, game: GameState, time):
        print("--" * 20)
        print(f"\n\n{player}\n\n -> mv.: {action}.\n\n{game.visualize(full_debug=debug)}\n")
        if debug:
            nonlocal max_eval, n_moves
            n_moves += 1
            if abs(game.evaluate(1, game.default_params)) > max_eval:
                max_eval = abs(game.evaluate(1, game.default_params))

        print("--" * 20)

        if pause:
            input("Press Enter to continue...")

        if game.is_terminal():
            if debug:
                print(f"{max_eval=}")
                print(f"{n_moves=}")
            if game.get_reward(1) == win:
                print("Game Over. Winner: P1")
            elif game.get_reward(1) == loss:
                print("Game Over. Winner: P2")
            else:
                print("Game Over. Draw")

    game, p1, p2 = init_game_and_players(game_key, game_params, p1_params, p2_params)
    try:
        reward = play_game_until_terminal(game, p1, p2, callback=callback)
        return reward
    finally:
        # Make sure that the statistics are printed even if an exception is raised (i.e. if the game is interrupted)
        p1.print_cumulative_statistics()
        p2.print_cumulative_statistics()


@log_exception_handler
def run_game_experiment(game_key: str, game_params: Dict[str, Any], p1_params: AIParams, p2_params: AIParams):
    """
    Run a game experiment with two AI players, and return detailed game information.

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

    times = []
    n_moves = 0
    # For the first move
    times.append(time.time())

    def callback(current_player, action, game, time):
        nonlocal n_moves
        times.append(time)
        print(
            f"Player {game.player}, action: {action}, time: {times[len(times) - 1] - times[len(times) - 2]:.1f}"
        )
        print(game.visualize())
        n_moves += 1

    start_time_total = time.time()
    result = play_game_until_terminal(game, p1, p2, callback=callback)
    total_time = time.time() - start_time_total

    # Calculate time intervals between moves for each player
    time_intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    odd_moves = time_intervals[::2]
    even_moves = time_intervals[1::2]

    avg_time_per_move = (sum(odd_moves) / len(odd_moves), sum(even_moves) / len(even_moves))

    if result == win:
        result = 1
        print(f"Game Over. Loser: P2 [{p2_params}]")
        print(f"Game Over. Winner: P1 [{p1_params}]")
    elif result == loss:
        result = 2
        print(f"Game Over. Loser: P1 [{p1_params}]")
        print(f"Game Over. Winner: P2 [{p2_params}]")
    else:
        result = 0
        print("Game Over. Draw []")

    p1.print_cumulative_statistics()
    p2.print_cumulative_statistics()

    return setup, total_time, avg_time_per_move, n_moves, result

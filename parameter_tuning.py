import math
import multiprocessing
from functools import partial
from itertools import product

from games.amazons import AmazonsGameState, evaluate as evaluate_amazons
from games.breakthrough import BreakthroughGameState, evaluate as evaluate_breakthrough
from run_games import mcts_play


def experiment(game_type, mcts_simulations, evaluation_function, num_games):
    experiment_name = f"{game_type}_simulations_{mcts_simulations}_eval_{evaluation_function.__name__}"
    print(f"Starting experiment: {experiment_name}")

    player1 = partial(mcts_play, num_simulations=mcts_simulations, evaluation_fn=evaluation_function)
    player2 = partial(mcts_play, num_simulations=mcts_simulations, evaluation_fn=evaluation_function)

    results = []
    for _ in range(num_games):
        state = BreakthroughGameState() if game_type == "breakthrough" else AmazonsGameState()
        while not state.is_terminal():
            action = player1(state) if state.player == 1 else player2(state)
            state = state.apply_action(action)
        results.append(state.get_reward())

    mean = sum(results) / num_games
    variance = sum((x - mean) ** 2 for x in results) / (num_games - 1)
    std_dev = math.sqrt(variance)
    conf_interval = 1.96 * std_dev / math.sqrt(num_games)

    return experiment_name, (mean, std_dev, conf_interval)


def run_parallel_experiments(game_types, mcts_simulations_list, evaluation_functions, num_games, parallelism):
    experiment_params = product(game_types, mcts_simulations_list, evaluation_functions)
    experiment_fn = partial(experiment, num_games=num_games)

    with multiprocessing.Pool(parallelism) as pool:
        results = pool.starmap(experiment_fn, experiment_params)

    for experiment_name, (mean, std_dev, conf_interval) in results:
        print(
            f"{experiment_name}: Mean = {mean}, Std Dev = {std_dev}, 95% Confidence Interval = +/- {conf_interval}"
        )


if __name__ == "__main__":
    game_types = ["breakthrough", "amazons"]
    mcts_simulations_list = [50, 100, 200]
    evaluation_functions = [evaluate_breakthrough, evaluate_amazons]
    num_games = 100
    parallelism = multiprocessing.cpu_count()

    run_parallel_experiments(game_types, mcts_simulations_list, evaluation_functions, num_games, parallelism)

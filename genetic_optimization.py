import csv
import datetime
from functools import partial
import multiprocessing
from operator import itemgetter
import random

from typing import Dict, List, Tuple, Any, Union
import gspread
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials

from ai.ai_player import AIPlayer
from games.gamestate import GameState
from run_games import AIParams, init_game_and_players

# Example Parameters for the genetic algorithm
population_size = 50
num_generations = 100
mutation_rate = 0.05


sheet = None


def setup_reporting(
    sheet_name: str,
    ai_param_ranges: Dict[str, Tuple[float, float]],
    eval_param_ranges: Dict[str, Tuple[float, float]],
) -> Tuple[Any, Any]:
    # Use credentials to create a client to interact with the Google Drive API
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("client_secret.json", scopes)
    client = gspread.authorize(creds)

    sheet = client.create(sheet_name).sheet1
    f = open(f"{sheet_name}.csv", "w", newline="")
    writer = csv.writer(f)
    csv_header = ["Generation"] + list(ai_param_ranges.keys()) + list(eval_param_ranges.keys()) + ["Fitness"]
    writer.writerow(csv_header)
    return sheet, writer


def report_results(
    sheet: Any,
    writer: Any,
    generation: int,
    best_individual: Dict[str, Dict[str, float]],
    best_individual_fitness: int,
) -> None:
    row = (
        [generation]
        + list(best_individual["ai"].values())
        + list(best_individual["eval"].values())
        + [best_individual_fitness]
    )
    sheet.append_row(row)
    writer.writerow(row)


def report_final_results(
    sheet: Any, writer: Any, best_overall_individual: Tuple[Dict[str, Dict[str, float]], int]
) -> None:
    row = (
        ["Final"]
        + list(best_overall_individual[0]["ai"].values())
        + list(best_overall_individual[0]["eval"].values())
        + [best_overall_individual[1]]
    )
    sheet.append_row(row)
    writer.writerow(row)


def genetic_algorithm(
    population_size: int,
    num_generations: int,
    mutation_rate: float,
    game_name: str,
    game_params: Dict[str, Any],
    player_name: str,
    eval_name: str,
    ai_param_ranges: Dict[str, Tuple[float, float]],
    eval_param_ranges: Dict[str, Tuple[float, float]],
    n_procs: int = 22,
    convergence_generations=5,
    tournament_size=5,
    elite_count=2,
):
    """
    Run the genetic algorithm to optimize the parameters of the AI and evaluation function.

    Args:
        population_size (int): The size of the population in each generation.
        num_generations (int): The number of generations to run the algorithm for.
        mutation_rate (float): The probability of each parameter being mutated.
        game_name (str): The name of the game to simulate.
        game_params (Dict[str, Any]): The parameters for the game.
        player_name (str): The name of the player AI.
        eval_name (str): The name of the evaluation function.
        ai_param_ranges (Dict[str, Tuple[float, float]]): A dictionary mapping parameter names to tuples defining the allowed range of that parameter.
        eval_param_ranges (Dict[str, Tuple[float, float]]): Similar to ai_param_ranges, but for the eval parameters.
        n_procs (int, optional): The number of processes to use for parallel computation. Defaults to 22.
        convergence_generations (int, optional): The number of generations over which to check for convergence. Defaults to 5.
        tournament_size (int, optional): The size of the tournaments used for selection. Defaults to 5.
        elite_count (int, optional): The number of elites to preserve between generations. Defaults to 2.

    Example:
        >>> ai_param_ranges = {'a': {'range': (0, 10), 'precision': 2}, 'b': {'range': (0, 1), 'precision': 3},}
        >>> eval_param_ranges = {'m': {'range': ((0, 10), (1, 10), (1, 2), (1, 4)), 'precision': (1, 2, 2, 3)},}
        >>> genetic_algorithm(50, 100, 0.05, 'TicTacToe', 'MiniMax', 'eval_func', ai_param_ranges, eval_param_ranges, tournament_size=3, elite_count=5, convergence_generations=10)
    """

    # Initialize population with random parameters for AI and eval function
    population = [
        {
            "ai": {
                param_name: round(random.uniform(r["range"][0], r["range"][1]), r["precision"])
                for param_name, r in ai_param_ranges.items()
            },
            "eval": {
                param_name: round(random.uniform(r["range"][0], r["range"][1]), r["precision"])
                for param_name, r in eval_param_ranges.items()
            },
        }
        for _ in range(population_size)
    ]

    # Create a new sheet for this experiment
    if game_params and "board_size" in game_params:
        game_name_str = game_name + str(game_params["board_size"])
    else:
        game_name_str = game_name
    sheet_name = (
        f"{game_name_str}_{player_name}_{eval_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    sheet, writer = setup_reporting(sheet_name, ai_param_ranges, eval_param_ranges)

    best_overall_individual = None
    best_fitnesses = []

    # Pre-fill the evaluate_fitness function with the game parameters
    partial_evaluate_fitness = partial(evaluate_fitness, game_name, game_params, player_name, eval_name)

    for generation in range(num_generations):
        print(f"Generation {generation}")

        # Randomly pair up individuals in the population
        random.shuffle(population)
        pairs = [(population[i], population[i + 1]) for i in range(0, len(population), 2)]

        # Evaluate the fitness of each pair of individuals in the population
        with multiprocessing.Pool(n_procs) as pool:
            results = pool.map(partial_evaluate_fitness, pairs)

        # Calculate the number of wins for each individual as fitness values
        fitnesses = [0] * len(population)
        for individual1, fitness1, individual2, fitness2 in results:
            fitnesses[population.index(individual1)] += fitness1
            fitnesses[population.index(individual2)] += fitness2

        scaled_fitnesses = scale_fitnesses(fitnesses)

        # Get the best individual and its fitness
        best_individual_index = fitnesses.index(max(fitnesses))
        best_individual = population[best_individual_index]

        if best_overall_individual is None or fitnesses[best_individual_index] > best_overall_individual[1]:
            best_overall_individual = (best_individual, fitnesses[best_individual_index])

        report_results(sheet, writer, generation, best_individual, fitnesses[best_individual_index])

        # Select individuals to reproduce based on their fitness
        # parents = select_parents(population, fitnesses)
        parents = select_parents_tournament(population, scaled_fitnesses, tournament_size)

        # Create the next generation through crossover and mutation
        population = create_next_generation(parents, mutation_rate, ai_param_ranges, eval_param_ranges)

        # Preserve the elites
        elites = sorted(zip(population, fitnesses), key=itemgetter(1), reverse=True)[:elite_count]
        for i in range(elite_count):
            population[i] = elites[i][0]

        # Convergence criterion
        best_fitnesses.append(max(fitnesses))
        if (
            len(best_fitnesses) > convergence_generations
            and len(set(best_fitnesses[-convergence_generations:])) == 1
        ):
            print("Converged.")
            break

    # At the end of all generations, write the best overall individual and its fitness
    report_final_results(sheet, writer, best_overall_individual)


def evaluate_fitness(
    game_name: str,
    game_params: Dict[str, Any],
    player_name: str,
    eval_name: str,
    pair: Tuple[Dict[str, Any], Dict[str, Any]],
) -> Tuple[Dict[str, Any], float, Dict[str, Any], float]:
    """
    Evaluate the fitness of two individuals by simulating a game.

    This function simulates a game with the given AIs and evaluation functions,
    using the parameters specified in the individuals' dictionaries.
    The fitness is the score of the game (1 for a win, 0 for a loss, 0.5 for a draw).

    Args:
        game_name (str): The name of the game to simulate.
        game_params (Dict[str, Any]): The parameters for the game.
        player_name (str): The name of the player AI.
        eval_name (str): The name of the evaluation function.
        pair (tuple): The pair of individuals to evaluate. Each individual is a dictionary with
                      "ai" and "eval" keys, each of which is a dictionary of parameter names to values.

    Returns:
        tuple: A pair consisting of the individuals and their fitness scores.
    """
    individual1, individual2 = pair

    # AI parameters for individuals
    ai_params1 = individual1["ai"]
    eval_params1 = individual1["eval"]
    ai_params2 = individual2["ai"]
    eval_params2 = individual2["eval"]

    # Create AI players with given parameters
    p1_params = AIParams(player_name, eval_name, ai_params1, eval_params1)
    p2_params = AIParams(player_name, eval_name, ai_params2, eval_params2)

    game, player1, player2 = init_game_and_players(game_name, game_params, p1_params, p2_params)

    # Play a game!
    current_player = player1
    while not game.is_terminal():
        # Get the best action for the current player
        action = current_player.best_action(game)

        # Apply the action to get the new game state
        game = game.apply_action(action)

        # Switch the current player
        current_player = player2 if current_player == player1 else player1

    # Return the winner (in view of p1)
    reward = game.get_reward(1)

    if reward == 0:
        # Draw
        return individual1, 0.5, individual2, 0.5
    elif reward == 1:
        # Individual1 wins
        return individual1, 1, individual2, 0
    else:
        # Individual2 wins
        return individual1, 0, individual2, 1


def scale_fitnesses(fitnesses):
    """
    Scales the fitness of each individual based on rank.

    Args:
        fitnesses (list): The fitness of each individual.

    Returns:
        list: The scaled fitnesses.

    Example:
        >>> fitnesses = [2, 1, 3]
        >>> scale_fitnesses(fitnesses)
        [2, 1, 3]
    """
    sorted_indices = np.argsort(fitnesses)[::-1]
    scaled_fitnesses = np.empty_like(fitnesses)
    for i in range(len(fitnesses)):
        # Assign scaled fitness based on rank
        scaled_fitnesses[sorted_indices[i]] = i + 1
    return scaled_fitnesses.tolist()


def select_parents_tournament(population, fitnesses, tournament_size):
    """
    Selects parents for the next generation using tournament selection.

    Args:
        population (list): The current population of individuals.
        fitnesses (list): The fitness of each individual in the population.
        tournament_size (int): The number of individuals in each tournament.

    Returns:
        list: The selected parents.

    Example:
        >>> population = [{'ai': {'a': 0.5}, 'eval': {'b': 0.5}}, {'ai': {'a': 0.1}, 'eval': {'b': 0.1}}]
        >>> fitnesses = [2, 1]
        >>> tournament_size = 2
        >>> select_parents_tournament(population, fitnesses, tournament_size)
        [{'ai': {'a': 0.5}, 'eval': {'b': 0.5}}, {'ai': {'a': 0.5}, 'eval': {'b': 0.5}}]
    """
    # Tournament selection
    parents = []
    for _ in range(len(population)):
        # Randomly select individuals for the tournament
        tournament_individuals = random.sample(list(zip(population, fitnesses)), tournament_size)
        # Select the best individual from the tournament
        winner = max(tournament_individuals, key=itemgetter(1))
        parents.append(winner[0])
    return parents


def select_parents_roulette(population: List[dict], fitnesses: List[float]) -> np.ndarray:
    """
    Select parents for the next generation based on their fitness.

    This function uses roulette wheel selection to probabilistically select individuals from the population to be parents
    for the next generation. Individuals with higher fitness have a higher chance of being selected.

    Args:
        population (List[dict]): The current population of individuals.
        fitnesses (List[float]): The fitness of each individual in the population.

    Returns:
        np.ndarray: The selected parents for the next generation.

    Example:
        >>> population = [{'ai': {'a': 1, 'b': 2}, 'eval': {'c': 3, 'd': 4}},
        ...               {'ai': {'a': 2, 'b': 3}, 'eval': {'c': 4, 'd': 5}},
        ...               {'ai': {'a': 3, 'b': 4}, 'eval': {'c': 5, 'd': 6}}]
        >>> fitnesses = [0.1, 0.2, 0.7]
        >>> select_parents(population, fitnesses)
        array([{'ai': {'a': 3, 'b': 4}, 'eval': {'c': 5, 'd': 6}},
               {'ai': {'a': 1, 'b': 2}, 'eval': {'c': 3, 'd': 4}},
               {'ai': {'a': 3, 'b': 4}, 'eval': {'c': 5, 'd': 6}}], dtype=object)
    """
    # Roulette wheel selection
    total_fitness = sum(fitnesses)
    selection_probs = [fitness / total_fitness for fitness in fitnesses]
    return np.random.choice(population, size=len(population), replace=True, p=selection_probs)


def create_next_generation(
    parents: List[dict],
    mutation_rate: float,
    ai_param_ranges: Dict[str, Tuple[float, float]],
    eval_param_ranges: Dict[str, Tuple[float, float]],
) -> List[dict]:
    """
    Create the next generation of individuals.

    This function creates the next generation by performing crossover and mutation on selected parents.

    Args:
        parents (List[dict]): The selected parents for the next generation.
        mutation_rate (float): The probability of each parameter being mutated.
        ai_param_ranges (dict): A dictionary mapping parameter names to tuples defining the allowed range of that parameter.
        eval_param_ranges (dict): Similar to ai_param_ranges, but for the eval parameters.

    Returns:
        List[dict]: The next generation of individuals.

    Example:
        >>> parents = [{'ai': {'a': 1, 'b': 2}, 'eval': {'c': 3, 'd': 4}},
        ...            {'ai': {'a': 2, 'b': 3}, 'eval': {'c': 4, 'd': 5}}]
        >>> ai_param_ranges = {'a': (0, 10), 'b': (0, 10)}
        >>> eval_param_ranges = {'c': (0, 10), 'd': (0, 10)}
        >>> create_next_generation(parents, 0.5, ai_param_ranges, eval_param_ranges)
        [{'ai': {'a': 1.567, 'b': 2.1}, 'eval': {'c': 2.998, 'd': 4.432}},
         {'ai': {'a': 2.203, 'b': 2.913}, 'eval': {'c': 3.829, 'd': 5.129}}]
    """
    next_generation = []
    for _ in range(len(parents) // 2):
        # Select two parents
        parent1, parent2 = random.sample(parents, 2)

        # Perform crossover
        child1, child2 = crossover(parent1, parent2)

        # Perform mutation
        child1 = mutate(child1, mutation_rate, ai_param_ranges, eval_param_ranges)
        child2 = mutate(child2, mutation_rate, ai_param_ranges, eval_param_ranges)

        next_generation.append(child1)
        next_generation.append(child2)

    return next_generation


def crossover(parent1: dict, parent2: dict) -> Tuple[dict, dict]:
    """
    Perform a single point crossover on two parent individuals.

    This function combines two parent individuals to create two child individuals,
    each inheriting some parameters from each parent. This is done separately for
    the "ai" parameters and the "eval" parameters.

    Args:
        parent1 (dict): The first parent individual.
                        Each individual is a dictionary with "ai" and "eval" keys,
                        each of which is a dictionary of parameter names to values.
        parent2 (dict): The second parent individual.

    Returns:
        tuple: A pair of child individuals, represented as dictionaries.

    Example:
        >>> parent1 = {'ai': {'a': 1, 'b': 2}, 'eval': {'c': 3, 'd': 4}}
        >>> parent2 = {'ai': {'a': 5, 'b': 6}, 'eval': {'c': 7, 'd': 8}}
        >>> crossover(parent1, parent2)
        ({'ai': {'a': 1, 'b': 2}, 'eval': {'c': 7, 'd': 8}}, {'ai': {'a': 5, 'b': 6}, 'eval': {'c': 3, 'd': 4}})
    """
    crossover_point = random.randint(1, len(parent1["ai"]) + len(parent1["eval"]) - 1)

    child1 = {}
    child2 = {}

    child1["ai"], child2["ai"] = crossover_dict(parent1["ai"], parent2["ai"], crossover_point)
    child1["eval"], child2["eval"] = crossover_dict(
        parent1["eval"], parent2["eval"], crossover_point - len(parent1["ai"])
    )

    return child1, child2


def crossover_dict(parent1: dict, parent2: dict, crossover_point: int) -> Tuple[dict, dict]:
    """
    Helper function for the crossover operation, for a single dictionary of parameters.

    This function takes two parent dictionaries and a crossover point, and returns
    two child dictionaries that inherit some keys from each parent.

    Args:
        parent1 (dict): The first parent dictionary.
        parent2 (dict): The second parent dictionary.
        crossover_point (int): The index at which to crossover the two parents.

    Returns:
        tuple: A pair of child dictionaries.

    Example:
        >>> parent1 = {'a': 1, 'b': 2, 'c': 3}
        >>> parent2 = {'a': 4, 'b': 5, 'c': 6}
        >>> crossover_dict(parent1, parent2, 2)
        ({'a': 1, 'b': 2, 'c': 6}, {'a': 4, 'b': 5, 'c': 3})
    """
    child1 = {}
    child2 = {}

    for i, key in enumerate(parent1):
        if i < crossover_point:
            child1[key] = parent1[key]
            child2[key] = parent2[key]
        else:
            child1[key] = parent2[key]
            child2[key] = parent1[key]

    return child1, child2


def mutate(
    individual: dict,
    mutation_rate: float,
    ai_param_ranges: Dict[str, Tuple[float, float]],
    eval_param_ranges: Dict[str, Tuple[float, float]],
) -> dict:
    """
    Mutate an individual's parameters with a certain probability.

    This function introduces variation into the population by applying small changes
    to the parameters of an individual with a certain probability, defined by the mutation rate.
    The function mutates the parameters of the individual in place, modifying the original dictionary.

    The parameters can be both single values or tuples, and the precision of mutation is defined
    for each parameter.

    Args:
        individual (dict): The individual to mutate.
                           Each individual is a dictionary with "ai" and "eval" keys,
                           each of which is a dictionary of parameter names to values.
        mutation_rate (float): The probability of each parameter being mutated.
        ai_param_ranges (dict): A dictionary mapping parameter names to dictionaries defining the allowed range and precision of that parameter.
        eval_param_ranges (dict): Similar to ai_param_ranges, but for the eval parameters.

    Returns:
        dict: The mutated individual.

    Example:
        >>> individual = {'ai': {'a': 1, 'b': 2}, 'eval': {'c': (3, 4), 'd': 5}}
        >>> ai_param_ranges = {'a': {'range': (0, 10), 'precision': 2}, 'b': {'range': (0, 10), 'precision': 2}}
        >>> eval_param_ranges = {'c': {'range': [(0, 10), (0, 10)], 'precision': [2, 2]}, 'd': {'range': (0, 10), 'precision': 2}}
        >>> mutate(individual, 0.5, ai_param_ranges, eval_param_ranges)
        {'ai': {'a': 1.12, 'b': 1.89}, 'eval': {'c': (2.94, 4.11), 'd': 5.06}}
    """
    individual["ai"] = _mutate_part(individual["ai"], mutation_rate, ai_param_ranges)
    individual["eval"] = _mutate_part(individual["eval"], mutation_rate, eval_param_ranges)

    return individual


def _mutate_part(part: dict, mutation_rate: float, param_ranges: Dict[str, dict]) -> dict:
    """
    Mutates a part of an individual.

    Args:
        part (dict): The part of the individual to mutate.
        mutation_rate (float): The probability of each parameter being mutated.
        param_ranges (dict): A dictionary mapping parameter names to dictionaries defining the allowed range and precision of that parameter.

    Returns:
        dict: The mutated part.
    """
    for key in part:
        if random.random() < mutation_rate:
            part[key] = _mutate_param(part[key], param_ranges[key])

    return part


def _mutate_param(
    param: Union[float, Tuple[float, ...]], param_info: dict
) -> Union[float, Tuple[float, ...]]:
    """
    Mutates a parameter of an individual.

    Args:
        param (float): The parameter to mutate.
        param_info (dict): A dictionary defining the allowed range and precision of the parameter.

    Returns:
        float or tuple of float: The mutated parameter.
    """
    return _mutate_value(param, param_info["range"], param_info["precision"])


def _mutate_value(value: float, value_range: Tuple[float, float], precision: int) -> float:
    """
    Mutates a value within a given range and precision.

    Args:
        value (float): The value to mutate.
        value_range (tuple of float): The allowed range of the value.
        precision (int): The number of decimal places to which to round the mutated value.

    Returns:
        float: The mutated value.
    """
    mutation = np.random.normal(0, (value_range[1] - value_range[0]) / 10)
    mutated_value = np.clip(value + mutation, value_range[0], value_range[1])

    return round(mutated_value, precision)

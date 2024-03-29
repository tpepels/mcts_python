import argparse
import csv
import datetime
import itertools
import json
import multiprocessing
import os
import random
import time
from functools import partial
from io import TextIOWrapper
from operator import itemgetter
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import jsonschema
from run_games import AIParams, init_game_and_players, play_game_until_terminal
from util import ErrorLogger, format_time, log_exception_handler

csv_f: TextIOWrapper = None
csv_writer = None
GLOBAL_N_PROCS = None  # set this value to override the number of cpu's to use for all experiments


@log_exception_handler
def setup_reporting(
    sheet_name: str,
    game_name: str,
    ai_param_ranges: Dict[str, Tuple[float, float]],
    eval_param_ranges: Dict[str, Tuple[float, float]],
):
    csv_header = (
        ["Generation"]
        + sorted(list(ai_param_ranges.keys()))
        + sorted(list(eval_param_ranges.keys()))
        + ["Fitness"]
    )
    global csv_f, csv_writer

    path = os.path.join(base_path, game_name)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f"{sheet_name}.csv")
    print(f"Writing to {path}")
    csv_f = open(path, "w", newline="")
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(csv_header)
    csv_f.flush()


def report_results(
    generation: int,
    best_individual: Dict[str, Dict[str, float]],
    best_individual_fitness: int,
) -> None:
    # Sort the keys so that values are reported in the correct order
    ai_params = [best_individual["ai"][key] for key in sorted(best_individual["ai"].keys())]
    eval_params = [best_individual["eval"][key] for key in sorted(best_individual["eval"].keys())]

    row = [generation] + ai_params + eval_params + [best_individual_fitness]

    csv_writer.writerow(row)
    csv_f.flush()


def report_final_results(best_overall_individual: Tuple[Dict[str, Dict[str, float]], int]) -> None:
    ai_params = [
        best_overall_individual[0]["ai"][key] for key in sorted(best_overall_individual[0]["ai"].keys())
    ]
    eval_params = [
        best_overall_individual[0]["eval"][key] for key in sorted(best_overall_individual[0]["eval"].keys())
    ]
    row = ["Final"] + ai_params + eval_params + [best_overall_individual[1]]

    csv_writer.writerow(row)
    csv_f.flush()


def generate_individual(param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Union[int, float]]:
    """
    Generate an individual with random parameters within the given ranges.

    Args:
        param_ranges (Dict[str, Tuple[float, float]]): A dictionary mapping parameter names to tuples defining the allowed range of that parameter.

    Returns:
        Dict[str, Union[int, float]]: The individual with randomly assigned parameters.
    """
    unsorted_dict = {
        param_name: random.randint(r["range"][0], r["range"][1])
        if r["precision"] == 0
        else round(random.uniform(r["range"][0], r["range"][1]), r["precision"])
        for param_name, r in param_ranges.items()
    }
    return {k: unsorted_dict[k] for k in sorted(unsorted_dict.keys())}


def create_pairs_against_opponent(population, fixed_opponent, games_per_pair):
    pairs = [(ind, fixed_opponent) for ind in population] * games_per_pair
    return pairs


def create_pairs_from_population(population, games_per_pair, debug=False):
    """
    Create pairs of individuals from the population. Each individual will play the other individual in the pair.

    Args:
        population (_type_): The population of individuals.
        games_per_pair (_type_): How many games each pair of individuals should play.
        debug (bool, optional): Whether to print debug statements. Defaults to False.

    Returns:
        list[tuple]: The pairs of individuals.
    """
    # Convert population of dictionaries into list of JSON strings
    pop_as_json = [json.dumps(ind, sort_keys=True) for ind in population]
    # Get unique individuals
    # unique_individuals = [json.loads(ind) for ind in list(set(pop_as_json))]
    unique_individuals = list(set(pop_as_json))
    pairs = list(itertools.combinations(unique_individuals, 2))
    total_games = games_per_pair * len(population)

    # Case 1: There are too few pairs, i.e. increase the number of games per pair while keeping the number of games played per pair the same
    while len(pairs) < total_games:
        if debug:
            print(f"Adding games to pairs, {len(pairs)} pairs, {len(unique_individuals)} unique individuals")
        pairs = pairs + pairs

    # Case 2: There are too many pairs. i.e. reduce the number of games per pair while keeping the number of games played per pair the same
    if len(pairs) > total_games:
        if debug:
            print(
                f"Removing games from pairs, {len(pairs)} pairs, {len(unique_individuals)} unique individuals"
            )
        random.shuffle(pairs)
        min_var = 10000
        best_pairs = None
        for _ in range(30):
            random.shuffle(pairs)
            new_pairs = pairs[:total_games]
            new_var = np.var([sum([ind in pair for pair in new_pairs]) for ind in unique_individuals])
            if new_var <= min_var:
                min_var = new_var
                best_pairs = new_pairs
        pairs = best_pairs

    if debug:
        print(
            f"Self-play between {len(pairs)} pairs, {len(unique_individuals)} unique individuals (wanted {total_games} games)"
        )
        print(
            f"number of pairings per indiv.:{[sum([pair.count(ind) for pair in pairs]) for ind in unique_individuals]}"
        )
    # Convert the json pairs back to dicts
    pairs = [(json.loads(pair1), json.loads(pair2)) for pair1, pair2 in pairs]
    return pairs


@log_exception_handler
def genetic_algorithm(
    population_size: int,
    num_generations: int,
    mutation_rate: float,
    game_name: str,
    player_name: str,
    ai_param_ranges: Optional[Dict[str, Tuple[float, float]]] = {},
    eval_param_ranges: Optional[Dict[str, Tuple[float, float]]] = {},
    game_params: Optional[Dict[str, Any]] = {},
    ai_static_params: Optional[Dict[str, Any]] = {},
    eval_static_params: Optional[Dict[str, Any]] = {},
    opponent_name: Optional[str] = None,
    fixed_opponent: Optional[dict] = {},
    best_as_opponent: bool = False,
    games_per_pair: int = 5,
    n_procs: int = 8,
    convergence_generations: int = 5,
    tournament_size: int = 5,
    elite_count: int = 4,
    draw_score: float = 0.125,
    default_eval_as_elite: dict = {},
    default_ai_as_elite: dict = {},
    debug: bool = True,
):
    """
    Run the genetic algorithm to optimize the parameters of the AI and evaluation function.

    This function uses a genetic algorithm to evolve a population of individuals over several generations,
    optimizing the parameters of the AI and evaluation function to maximize performance in the given game.

    Args:
        population_size (int): The size of the population in each generation.
        num_generations (int): The number of generations to run the algorithm for.
        mutation_rate (float): The probability of each parameter being mutated.
        game_name (str): The name of the game to simulate.
        game_params (Dict[str, Any]): The parameters for the game.
        player_name (str): The name of the player AI.
        ai_param_ranges (Dict[str, Tuple[float, float]]): A dictionary mapping parameter names to tuples defining the allowed range of that parameter.
        eval_param_ranges (Dict[str, Tuple[float, float]]): Similar to ai_param_ranges, but for the eval parameters.
        ai_static_params (Dict[str, Any], optional): The static AI parameters (i.e. the parameters that are the same for all indivuduals). Defaults to {}.
        eval_static_params (Dict[str, Any], optional): The static evaluation parameters (i.e. the parameters that are the same for all individuals). Defaults to {}.
        n_procs (int, optional): The number of processes to use for parallel computation. Defaults to 8.
        convergence_generations (int, optional): The number of generations over which to check for convergence. Defaults to 5.
        tournament_size (int, optional): The size of the tournaments used for selection. Defaults to 5.
        elite_count (int, optional): The number of elites to preserve between generations. Defaults to 2.
        draw_score (float, optional): The score assigned for a draw. Defaults to 0.25.
        default_eval_as_elite: dict = {},
        default_ai_as_elite: dict = {},
        debug (bool, optional): Flag to enable debug outputs. Defaults to False.

    Example:
        Any parameters with 0 precision will be converted to integers.
        >>> ai_param_ranges = {'a': {'range': (0, 10), 'precision': 2}, 'b': {'range': (0, 1), 'precision': 3},}
        >>> eval_param_ranges = {'m': {'range': ((0, 10), (1, 10), (1, 2), (1, 4)), 'precision': (1, 2, 2, 3)},}
    """
    assert (
        ai_param_ranges or eval_param_ranges
    ), "One range of parameters must be given: either ai params or eval params"

    global GLOBAL_N_PROCS
    if GLOBAL_N_PROCS is not None:
        n_procs = GLOBAL_N_PROCS
    print(f"*** Using {n_procs} processes ***")

    if debug:
        start_time = time.time()

    if population_size % 2 != 0:
        raise ValueError("Population size must be even")

    # Initialize population with random parameters for AI and eval function
    population = [
        {
            "ai": generate_individual(ai_param_ranges),
            "eval": generate_individual(eval_param_ranges),
        }
        for _ in range(population_size)
    ]

    # * Add the default elite individuals to the population
    if default_eval_as_elite != {} and default_ai_as_elite == {}:
        for _ in range(elite_count):
            population.append({"ai": generate_individual(ai_param_ranges), "eval": default_eval_as_elite})
    if default_ai_as_elite != {} and default_eval_as_elite == {}:
        for _ in range(elite_count):
            population.append({"ai": default_ai_as_elite, "eval": generate_individual(eval_param_ranges)})
    if default_ai_as_elite != {} and default_eval_as_elite != {}:
        for _ in range(elite_count):
            population.append({"ai": default_ai_as_elite, "eval": default_eval_as_elite})

    best_overall_individual = None
    best_individuals = []
    best_fitnesses = []

    # Pre-fill the evaluate_fitness function with the game parameters
    partial_evaluate_fitness = partial(
        evaluate_fitness,
        game_name,
        game_params,
        player_name,
        opponent_name if opponent_name is not None else player_name,
        ai_static_params,
        eval_static_params,
        draw_score,
        base_path,
    )
    gen_times = []
    for generation in range(num_generations):
        if debug:
            print("." * 30 + time.strftime("%H:%M:%S", time.localtime()) + "." * 30)
            print(f"Starting generation {generation}")

        if opponent_name is None:
            pairs = create_pairs_from_population(population, games_per_pair, debug=debug)
        else:
            pairs = create_pairs_against_opponent(population, fixed_opponent, games_per_pair)
        #
        gen_start_time = time.time()
        if debug:
            print(f"Created {len(pairs)} pairs of individuals.")

        # Evaluate the fitness of each pair of individuals in the population
        with multiprocessing.Pool(n_procs) as pool:
            results = pool.map(partial_evaluate_fitness, pairs)

        # Create a new sheet for this experiment, do this after the first results so we don't create a lot of unused sheets
        if generation == 0:
            if game_params and "board_size" in game_params:
                game_name_str = game_name + str(game_params["board_size"])
            else:
                game_name_str = game_name
            sheet_name = f"{game_name_str}_{player_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            # This prepares the csv file and google sheet
            setup_reporting(sheet_name, game_name, ai_param_ranges, eval_param_ranges)

        # Calculate the number of wins for each individual as fitness values
        fitnesses = [0.0] * len(population)  # Initialize the fitnesses list
        for individual1, fitness1, individual2, fitness2 in results:
            # Find indices of the individuals
            index1 = population.index(individual1)
            fitnesses[index1] += fitness1

            # Interplay, so we need to also consider the opponent
            if opponent_name is None:
                index2 = population.index(individual2)
                fitnesses[index2] += fitness2

        if debug:
            # print the fitnesses per individual:
            for fitness in fitnesses:
                print(f"fitness: {fitness}")

        if debug:
            print("." * 30 + time.strftime("%H:%M:%S", time.localtime()) + "." * 30)
            print(f"Finished fitness calculation, took {format_time(int(time.time() - gen_start_time))}")

        # Get the best individual and its fitness
        best_individual_index = fitnesses.index(max(fitnesses))
        best_individual = population[best_individual_index]

        # * Select the best individual as the opponent for the next generation
        if best_as_opponent:
            fixed_opponent = best_individual
            # ! If player_name == opponent_name, nothing changes
            # ! If player_name != opponent_name, we need to be sure that we are playing against the same algorithm otherwise the parameters don't make any sense
            opponent_name = player_name

        # Append the best individual of each generation to the best_individuals list
        best_individuals.append(json.dumps(best_individual))
        best_fitnesses.append(max(fitnesses))

        if debug:
            print(f"Best individual: {best_individual}")
            print(f"Best fitness: {fitnesses[best_individual_index]}")

        if best_overall_individual is None or fitnesses[best_individual_index] > best_overall_individual[1]:
            best_overall_individual = (best_individual, fitnesses[best_individual_index])

        report_results(generation, best_individual, fitnesses[best_individual_index])

        # * Scaled fitness goes from best: 1 to worst: n
        scaled_fitnesses = scale_fitnesses(fitnesses)
        # Select individuals to reproduce based on their fitness
        parents = select_parents_tournament(population, scaled_fitnesses, tournament_size)
        if debug:
            print(scaled_fitnesses)
            print()
            i = 1
            for individual in parents:
                print(f"parent {i}: {individual}")
                i += 1
        # Preserve the elites
        potential_elites = sorted(zip(population, scaled_fitnesses), key=itemgetter(1))
        # Get the unique elites / this function preserves the order of the elites
        unique_elites = get_unique_individuals([elite[0] for elite in potential_elites])
        # Create the next generation through crossover and mutation
        population = create_next_generation(parents, mutation_rate, ai_param_ranges, eval_param_ranges)

        if debug:
            # print the population info
            print(f"* Population size: {len(population)}")
            print(f"* Number of unique individuals in previous population: {len(unique_elites)}")
            print(f"* Inserted {elite_count} elites into the population.")
            i = 1
            for individual in population:
                print(f"indiv {i}: {individual}")
                i += 1
            print(".." * 30)
            i = 1
            for individual in unique_elites[:elite_count]:
                print(f"elite {i}: {individual}")
                i += 1

        # Draw elites (the resulting number can be lower than elite_count if there are not enough unique elites)
        elites = unique_elites[:elite_count]
        replace_indices = set()
        # Get unique indices to be replaced
        while len(replace_indices) < len(elites):
            replace_index = random.randint(0, len(population) - 1)
            replace_indices.add(replace_index)

        # Replace individuals at selected indices with elites
        for replace_index, elite in zip(replace_indices, unique_elites[:elite_count]):
            population[replace_index] = elite

        # Convergence criterion
        if len(best_individuals) > convergence_generations:
            # The same individual was the best for the pas few generations
            if len(set(best_individuals[-convergence_generations:])) == 1:
                print(f"Converged at generation {generation}.")
                break

            # Extract the fitness values of the last 'convergence_generations'
            last_generations_fitness = best_fitnesses[-convergence_generations:]
            # Check if none of the fitness values increased
            if not any(x < y for x, y in zip(last_generations_fitness, last_generations_fitness[1:])):
                print(f"Converged at generation {generation}.")
                break
        elif debug:
            unique_individuals = len(set(best_individuals[-convergence_generations:]))
            print(
                f"# of unique individuals values in the last {convergence_generations} generations: {unique_individuals}"
            )

        if debug:
            gen_end_time = time.time()
            gen_time = gen_end_time - gen_start_time
            gen_times.append(gen_time)
            average_gen_time = sum(gen_times) / len(gen_times)
            remaining_generations = num_generations - (generation + 1)
            estimated_remaining_time = average_gen_time * remaining_generations

            print(f"    Finished generation {generation}, took {format_time(int(gen_time))}")
            print(f"    Estimated remaining time: {format_time(int(estimated_remaining_time))}")

    if debug:
        print("--" * 40)
        print(f"Finished all generations, took {format_time(int(time.time() - start_time))}")

    # At the end of all generations, write the best overall individual and its fitness
    report_final_results(best_overall_individual)


def get_unique_individuals(individuals: List[dict]) -> List[dict]:
    """
    Convert a list of individuals into a unique list of individuals.
    This function preserves the order of the individuals.

    Args:
        individuals (List[dict]): The list of individuals.

    Returns:
        List[dict]: The list of unique individuals.
    """
    # Convert the list of individuals to JSON strings
    individuals_as_json = [json.dumps(ind, sort_keys=True) for ind in individuals]

    # Get the unique individuals as JSON strings using dict to maintain order
    unique_individuals_as_json = list(dict.fromkeys(individuals_as_json))

    # Convert the unique individuals back to dictionaries
    unique_individuals = [json.loads(ind) for ind in unique_individuals_as_json]

    return unique_individuals


def evaluate_fitness(
    game_name: str,
    game_params: Dict[str, Any],
    player_name: str,
    opponent_name: str,
    ai_static_params: Dict[str, Any],
    eval_static_params: Dict[str, Any],
    draw_score: float,
    base_path: str,
    pair: Tuple[Dict[str, Any], Dict[str, Any]],
) -> Tuple[Dict[str, Any], float, Dict[str, Any], float]:
    """
    Evaluate the fitness of two individuals by simulating a game.

    This function simulates a game with the given AIs and evaluation functions,
    using the parameters specified in the individuals' dictionaries.
    The fitness is the score of the game (1 for a win, 0 for a loss, and a draw_score for a draw).

    Args:
        game_name (str): The name of the game to simulate.
        game_params (Dict[str, Any]): The parameters for the game.
        player_name (str): The name of the player AI.
        ai_static_params (Dict[str, Any]): The static parameters (i.e. the parameters that are the same for all indivuduals).
        eval_static_params (Dict[str, Any]): The static parameters (i.e. the parameters that are the same for all indivuduals).
        draw_score (float): The score assigned for a draw.
        pair (tuple): The pair of individuals to evaluate. Each individual is a dictionary with
                    "ai" and "eval" keys, each of which is a dictionary of parameter names to values.

    Returns:
        tuple: A pair consisting of the individuals and their fitness scores.
    """
    individual1, individual2 = pair

    # Merge the static parameters and the individual's parameters
    ai_params1 = {**ai_static_params, **individual1["ai"]}
    eval_params1 = {**eval_static_params, **individual1["eval"]}
    ai_params2 = {**ai_static_params, **individual2["ai"]}
    eval_params2 = {**eval_static_params, **individual2["eval"]}

    # Assert that nothing is None:
    assert ai_static_params is not None, f"ai_params1 is None in evaluate_fitness {pair=}"
    assert eval_static_params is not None, f"eval_params1 is None in evaluate_fitness {pair=}"
    assert individual1["ai"] is not None, f"individual1[ai] is None in evaluate_fitness {pair=}"
    assert individual1["eval"] is not None, f"individual1[eval] is None in evaluate_fitness {pair=}"
    assert individual2["ai"] is not None, f"individual2[ai] is None in evaluate_fitness {pair=}"
    assert individual2["eval"] is not None, f"individual2[eval] is None in evaluate_fitness {pair=}"

    n_moves = 0

    def callback(player, action, game, time):
        nonlocal n_moves
        n_moves += 1
        print(f"Player {player} - chosen action: {action}.\nCurrent game state:\n{game.visualize()}")
        if game.is_terminal():
            result = game.get_result_tuple()
            # def get_result_tuple(self) -> cython.tuple:
            #     if self.winner == 1:
            #         return (1.0, 0.0)
            #     elif self.winner == 2:
            #         return (0.0, 1.0)

            #     return (0.5, 0.5):
            print(f"{n_moves} moves total.")
            if result[0] == 1:
                print("Game Over. Winner: P1")
            elif result[1] == 1:
                print("Game Over. Winner: P2")
            else:
                print("Game Over. Draw")

    with ErrorLogger(play_game_until_terminal, log_dir=os.path.join(base_path, "log", "game_error")):
        # Create AI players with given parameters
        ai1_params = AIParams(
            ai_key=player_name,
            max_player=1,
            eval_params=eval_params1,
            ai_params=ai_params1,
            game_name=game_name,
        )
        ai2_params = AIParams(
            ai_key=opponent_name,
            max_player=2,
            eval_params=eval_params2,
            ai_params=ai_params2,
            game_name=game_name,
        )
        game, player1, player2 = init_game_and_players(game_name, game_params, ai1_params, ai2_params)
        play_game_until_terminal(game, player1, player2, callback=callback)
        game_result = game.get_result_tuple()
        n_moves = 0
        # Create AI players with given parameters for the swapped seats game
        ai1_params = AIParams(
            ai_key=player_name,
            max_player=2,
            eval_params=eval_params1,
            ai_params=ai_params1,
            game_name=game_name,
        )
        ai2_params = AIParams(
            ai_key=opponent_name,
            max_player=1,
            eval_params=eval_params2,
            ai_params=ai_params2,
            game_name=game_name,
        )
        # This method does not assign any order to the players, it just initializes them
        game, player1, player2 = init_game_and_players(game_name, game_params, ai1_params, ai2_params)
        # For this function, order does matter, so we swap the players
        play_game_until_terminal(game, player2, player1, callback=callback)
        game_result_sw = game.get_result_tuple()

    res_1 = get_game_result(game_result, draw_score)
    res_2 = get_game_result(game_result_sw, draw_score)

    individual1_res = res_1[0] + res_2[1]
    individual2_res = res_1[1] + res_2[0]

    return individual1, individual1_res, individual2, individual2_res


def get_game_result(game_result: tuple, draw_score: float) -> Tuple[float, float]:
    if game_result == (0.5, 0.5):
        # Draw
        return draw_score, draw_score
    else:
        return game_result


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


def select_parents_tournament(population, fitnesses_ranks, tournament_size):
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
    pop_fitness = list(zip(population, fitnesses_ranks))
    for _ in range(len(population)):
        # Randomly select individuals for the tournament
        tournament_individuals = random.sample(pop_fitness, tournament_size)
        # Select the best individual from the tournament (i.e. with the lowest fitness rank)
        winner = min(tournament_individuals, key=itemgetter(1))
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

    if not all(key in parent for key in ["ai", "eval"] for parent in [parent1, parent2]):
        raise ValueError("Both parents must have 'ai' and 'eval' keys.")

    child1 = {}
    child2 = {}

    if len(parent1["ai"]) > 1:
        crossover_point_ai = random.randint(1, len(parent1["ai"]) - 1)
        child1["ai"], child2["ai"] = crossover_dict(parent1["ai"], parent2["ai"], crossover_point_ai)
    else:
        child1["ai"] = parent1["ai"]
        child2["ai"] = parent2["ai"]

    if len(parent2["eval"]) > 1:
        crossover_point_eval = random.randint(1, len(parent1["eval"]) - 1)
        child1["eval"], child2["eval"] = crossover_dict(
            parent1["eval"], parent2["eval"], crossover_point_eval
        )
    else:
        child1["eval"] = parent1["eval"]
        child2["eval"] = parent2["eval"]

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
    tot_range = param_info["range"][1] - param_info["range"][0]
    mutation = np.random.normal(0, tot_range / 10)
    mutated_value = np.clip(param + mutation, param_info["range"][0], param_info["range"][1])

    return round(mutated_value, None if param_info["precision"] == 0 else param_info["precision"])


def run_experiments_from_file(file_path: str):
    """
    Run experiments from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing the experiments.
    """
    with open(file_path, "r") as f:
        experiments = json.load(f)

    start_time = time.time()
    total_experiments = len(experiments)
    times = []
    experiment: dict

    for i, experiment in enumerate(experiments):
        if experiment.get("status", "incomplete") == "complete":
            print(f"Experiment {i+1} already completed. Skipping...")
            continue

        if "status" in experiment:  # Don't pass the status parameter to the function
            experiment.pop("status")

        exp_start_time = time.time()

        # Creating a name and description for the experiment
        experiment_name = f"Experiment {i+1} - Game: {experiment.get('game_name', 'Unknown')} Player: {experiment.get('player_name', 'Unknown')}"
        print(f"Running {experiment_name} ({i+1}/{total_experiments})...")

        try:
            validate_experiment_config(experiment)
            genetic_algorithm(**experiment)
            experiment["status"] = "complete"
        except (jsonschema.exceptions.ValidationError, Exception) as e:
            print(f"An error occurred with {experiment_name}: {e}")
            print(traceback.format_exc())
            print(f"{experiment=}")
            experiment["status"] = "error"
            continue
        finally:
            # Always write the status back to the file, whether the experiment completed or failed
            with open(file_path, "w") as f:
                json.dump(experiments, f, indent=4)

        # Calculate time taken for the experiment and append it to times list
        exp_end_time = time.time()
        exp_time = exp_end_time - exp_start_time
        times.append(exp_time)

        # Estimate remaining time based on average time per experiment
        avg_time_per_experiment = sum(times) / len(times)
        estimated_remaining_time = avg_time_per_experiment * (total_experiments - (i + 1))

        print(f"Finished {experiment_name} in {format_time(exp_time)}.")
        print(f"Estimated remaining time: {format_time(estimated_remaining_time)}.\n")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Finished all experiments in {format_time(total_time)}.")


def validate_experiment_config(config):
    """
    Validate an experiment configuration dictionary against a schema.
    """
    # Define the schema for the configuration
    schema = {
        "type": "object",
        "properties": {
            "population_size": {"type": "integer"},
            "num_generations": {"type": "integer"},
            "mutation_rate": {"type": "number"},
            "game_name": {"type": "string"},
            "game_params": {"type": "object"},
            "player_name": {"type": "string"},
            "ai_param_ranges": {"type": "object"},
            "eval_param_ranges": {"type": "object"},
            "ai_static_params": {"type": "object"},
            "eval_static_params": {"type": "object"},
            "n_procs": {"type": "integer"},
            "convergence_generations": {"type": "integer"},
            "tournament_size": {"type": "integer"},
            "elite_count": {"type": "integer"},
            "draw_score": {"type": "number"},
            "default_eval_as_elite": {"type": "object"},
            "default_ai_as_elite": {"type": "object"},
            "debug": {"type": "boolean"},
            "status": {"type": "string"},
        },
        "required": ["population_size", "num_generations", "mutation_rate", "game_name", "player_name"],
    }

    # Validate the configuration
    jsonschema.validate(config, schema)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments from a JSON file.")
    parser.add_argument("file", help="The path to the JSON file containing the experiments.")
    parser.add_argument("--n_procs", type=int, help="The number of processors to use.")
    parser.add_argument("--base_path", type=str, default=".", help="Base directory to create log files.")
    args = parser.parse_args()

    base_path = os.path.join(args.base_path, os.path.splitext(os.path.basename(args.file))[0])
    print("Base path:", base_path)

    # Validate and create the base directory for logs if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Update the global variable based on the argument
    GLOBAL_N_PROCS = args.n_procs

    run_experiments_from_file(args.file)

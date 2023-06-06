import datetime
from operator import itemgetter
import numpy as np
import random
import gspread

from oauth2client.service_account import ServiceAccountCredentials
from multiprocessing import Pool
from functools import partial
from ai.ai_player import AIParams
from run_games import AIParams
from run_games import player_dict, game_dict

# Example Parameters for the genetic algorithm
population_size = 50
num_generations = 100
mutation_rate = 0.05

# Use credentials to create a client to interact with the Google Drive API
scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("client_secret.json", scopes)
client = gspread.authorize(creds)
sheet = None


def genetic_algorithm(
    population_size,
    num_generations,
    mutation_rate,
    game_name,
    player_name,
    eval_name,
    param_ranges,
    n_procs=22,
):
    # Initialize population with random parameters
    population = [[random.uniform(r[0], r[1]) for r in param_ranges] for _ in range(population_size)]

    # Create a new sheet for this experiment
    sheet_name = f"{game_name}_{player_name}_{eval_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sheet = client.create(sheet_name).sheet1

    # Create a new CSV file for this experiment
    with open(f"{sheet_name}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Generation", "Param1", "Param2", "Param3", "Param4", "Fitness"])

        for generation in range(num_generations):
            print(f"Generation {generation}")

            # Evaluate the fitness of each individual in the population
            with Pool(n_procs) as pool:
                fitnesses = pool.starmap(
                    partial(evaluate_fitness, game_name, player_name, eval_name), population
                )

            # Write the best individual and its fitness to the Google Sheet
            best_individual = max(zip(population, fitnesses), key=itemgetter(1))
            sheet.append_row([generation, *best_individual[0], best_individual[1]])

            # Write the best individual and its fitness to the CSV file
            writer.writerow([generation, *best_individual[0], best_individual[1]])

            # Select individuals to reproduce based on their fitness
            parents = select_parents(population, fitnesses)

            # Create the next generation through crossover and mutation
            population = create_next_generation(parents, mutation_rate, param_ranges)

    # At the end of all generations, write the best overall individual and its fitness to the Google Sheet
    best_individual = max(zip(population, fitnesses), key=itemgetter(1))
    sheet.append_row(["Final", *best_individual[0], best_individual[1]])

    # Write the best overall individual and its fitness to the CSV file
    writer.writerow(["Final", *best_individual[0], best_individual[1]])


def evaluate_fitness(game_name, player_name, eval_name, params):
    # TODO Deze functie klopt nog niet, hij moet init_game_and_players gebruiken
    # TODO De return klopt ook niet omdat het in view van p1 is

    # Get the class from the class name
    game_class = game_dict[game_name]
    player_class = player_dict[player_name]

    # Create AI players with given parameters
    ai_params_dict = {f"param{i}": value for i, value in enumerate(params)}

    player1 = player_class(game_class(), 1, 3, AIParams(player_name, eval_name, ai_params_dict, None))
    player2 = player_class(game_class(), 2, 3, AIParams(player_name, eval_name, ai_params_dict, None))

    # Have the AI players play a game against each other
    winner = play_game(player1, player2)

    # Return the fitness of the parameters
    return 1 if winner == 1 else 0


def create_next_generation(parents, mutation_rate, param_ranges):
    next_generation = []
    for _ in range(len(parents) // 2):
        # Select two parents
        parent1, parent2 = random.sample(parents, 2)

        # Perform crossover
        child1, child2 = crossover(parent1, parent2)

        # Perform mutation
        child1 = mutate(child1, mutation_rate, param_ranges)
        child2 = mutate(child2, mutation_rate, param_ranges)

        next_generation.append(child1)
        next_generation.append(child2)

    return next_generation


def select_parents(population, fitnesses):
    # Roulette wheel selection
    total_fitness = sum(fitnesses)
    selection_probs = [fitness / total_fitness for fitness in fitnesses]
    return np.random.choice(population, size=len(population), replace=True, p=selection_probs)


def crossover(parent1, parent2):
    # Single point crossover
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(individual, mutation_rate, param_ranges):
    # Randomly change each parameter with a small probability
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] += np.random.normal(0, (param_ranges[i][1] - param_ranges[i][0]) / 10)
            individual[i] = np.clip(individual[i], param_ranges[i][0], param_ranges[i][1])
    return individual


def play_game(player1, player2):
    current_player = player1
    while not current_player.state.is_terminal():
        # Get the best action for the current player
        action = current_player.best_action()

        # Apply the action to get the new game state
        new_state = current_player.state.apply_action(action)

        # Update the game state for each player
        player1.state = new_state
        player2.state = new_state

        # Switch the current player
        current_player = player2 if current_player == player1 else player1

    # Return the winner
    return current_player.state.get_reward()

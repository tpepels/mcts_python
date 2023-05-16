import random
import numpy as np

from games.gamestate import (
    GameState,
)
from ai.ai_player import AIPlayer
from ai.alpha_beta import AlphaBetaPlayer

# Parameters for the genetic algorithm
population_size = 50
num_generations = 100
mutation_rate = 0.05


def genetic_algorithm(population_size, num_generations, mutation_rate):
    # Initialize population with random parameters
    population = [np.random.rand(4) for _ in range(population_size)]

    for generation in range(num_generations):
        print(f"Generation {generation}")

        # Evaluate the fitness of each individual in the population
        fitnesses = [evaluate_fitness(individual) for individual in population]

        # Select individuals to reproduce based on their fitness
        parents = select_parents(population, fitnesses)

        # Create the next generation through crossover and mutation
        population = create_next_generation(parents, mutation_rate)

    # Return the individual with the highest fitness in the final population
    return max(population, key=evaluate_fitness)


def evaluate_fitness(parameters):
    # Create AI players with given parameters
    player1 = AlphaBetaPlayer(GameState(), 1, 3, lambda state, player: evaluate(state, player, parameters))
    player2 = AlphaBetaPlayer(GameState(), 2, 3, lambda state, player: evaluate(state, player, parameters))

    # Have the AI players play a game against each other
    winner = play_game(player1, player2)

    # Return the fitness of the parameters
    if winner == 1:
        return 1
    else:
        return 0


def select_parents(population, fitnesses):
    # Roulette wheel selection
    total_fitness = sum(fitnesses)
    selection_probs = [fitness / total_fitness for fitness in fitnesses]
    return np.random.choice(population, size=len(population), p=selection_probs)


def create_next_generation(parents, mutation_rate):
    next_generation = []
    for _ in range(len(parents) // 2):
        # Select two parents
        parent1, parent2 = random.sample(list(parents), 2)

        # Perform crossover
        child1, child2 = crossover(parent1, parent2)

        # Perform mutation
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)

        next_generation.append(child1)
        next_generation.append(child2)

    return next_generation


def crossover(parent1, parent2):
    # Single point crossover
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2


def mutate(individual, mutation_rate):
    # Randomly change each parameter with a small probability
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] += np.random.normal(0, 0.1)
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


# Run the genetic algorithm
best_parameters = genetic_algorithm(population_size, num_generations, mutation_rate)

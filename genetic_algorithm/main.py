# Task 10. Implement Genetic Algorithm (GA) Optimization
import random
import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# Selection Mechanisms
# Tournament selection involves selecting a subset of individuals randomly and choosing the best among them to be a parent.
def tournament_selection(population, fitnesses, problem, tournament_size=3):
    selected = []
    population_size = len(population)
    for _ in range(population_size):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        if problem == 'rastrigin':
            winner = min(tournament, key=lambda x: x[1])
        else:
            winner = max(tournament, key=lambda x: x[1])  # Assuming minimization
        selected.append(winner[0])
    return selected

# Roulette wheel selection assigns a probability to each individual based on its fitness.
# Individuals with better fitness have a higher chance of being selected.
def roulette_wheel_selection(population, fitnesses, problem):
    selected = []
    population_size = len(population)
    if problem == 'rastrigin':
        max_fitness = max(fitnesses)
        adjusted_fitnesses = [max_fitness - f + 1e-6 for f in fitnesses]  # Minimization
    else:
        adjusted_fitnesses = fitnesses.copy()  # Maximization
    total_fitness = sum(adjusted_fitnesses)
    probabilities = [f / total_fitness for f in adjusted_fitnesses]
    for _ in range(population_size):
        r = random.random()
        cumulative = 0
        for individual, prob in zip(population, probabilities):
            cumulative += prob
            if r <= cumulative:
                selected.append(individual)
                break
    return selected

# Crossover Techniques
# Single-point crossover splits two parents at a random point and exchanges the tail segments to create two offspring.
def single_point_crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Two-point crossover selects two points and exchanges the segments between them.
def two_point_crossover(parent1, parent2):
    if len(parent1) < 2:
        return parent1.copy(), parent2.copy()
    point1 = random.randint(1, len(parent1) - 2)
    point2 = random.randint(point1 + 1, len(parent1) - 1)
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return child1, child2

# Uniform crossover selects genes from each parent with a fixed probability.
def uniform_crossover(parent1, parent2, gene_exchange_prob=0.5):
    child1 = []
    child2 = []
    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < gene_exchange_prob:
            child1.append(gene2)
            child2.append(gene1)
        else:
            child1.append(gene1)
            child2.append(gene2)
    return child1, child2

# Mutation Operators
# Gaussian mutation adds a random value drawn from a Gaussian distribution to each gene.
def gaussian_mutation(individual, mutation_rate=0.1, sigma=0.1, lower_bound=-5.12, upper_bound=5.12):
    mutated = []
    for gene in individual:
        if random.random() < mutation_rate:
            gene += random.gauss(0, sigma)
            gene = max(lower_bound, min(upper_bound, gene))
        mutated.append(gene)
    return mutated

# Flip-bit mutation flips the value of a gene with a fixed probability.
def flip_bit_mutation(individual, mutation_rate=0.01):
    mutated = []
    for gene in individual:
        if random.random() < mutation_rate:
            mutated.append(1 - gene)
        else:
            mutated.append(gene)
    return mutated

# Encoding for Knapsack
def initialize_knapsack_population(pop_size, num_items):
    return [[random.randint(0,1) for _ in range(num_items)] for _ in range(pop_size)]

# Fitness Functions
def rastrigin(individual):
    A = 10
    n = len(individual)
    if n < 10:
        print("Use at least 10 dimensions in Rastrigin function.")
        return -1
    return A * len(individual) + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in individual])

def knapsack_fitness(individual, values, weights, max_weight):
    total_value = 0
    total_weight = 0
    for gene, value, weight in zip(individual, values, weights):
        if gene == 1:
            total_value += value
            total_weight += weight
    if total_weight > max_weight:
        return 0  # Invalid solution
    else:
        return total_value

# Genetic Algorithm
def genetic_algorithm(
    problem='rastrigin',
    pop_size=100,
    generations=100,
    crossover_rate=0.8,
    mutation_rate=0.1,
    selection_method='tournament',
    crossover_method='single_point',
    mutation_method='gaussian',
    num_dimensions=10,
    knapsack_values=None,
    knapsack_weights=None,
    knapsack_max_weight=None,
    no_improvement_generations=10  # For convergence detection
):
    # Initialize population
    if problem == 'rastrigin':
        population = [ [random.uniform(-5.12, 5.12) for _ in range(num_dimensions)] for _ in range(pop_size) ]
    elif problem == 'knapsack':
        population = initialize_knapsack_population(pop_size, len(knapsack_values))
    else:
        raise ValueError("Unsupported problem type.")

    best_fitness = float('inf') if problem == 'rastrigin' else 0
    best_individual = None

    # Lists to store fitness data for plotting
    best_fitness_history = []
    average_fitness_history = []
    std_fitness_history = []

    generations_to_converge = generations  # Initialize with max generations

    # Convergence tracking
    no_improvement_counter = 0

    for gen in range(generations):
        # Evaluate fitness
        if problem == 'rastrigin':
            fitnesses = [rastrigin(ind) for ind in population]
            current_best = min(fitnesses)
            current_best_individual = population[fitnesses.index(current_best)]
            current_average = sum(fitnesses) / len(fitnesses)
            current_std = np.std(fitnesses)
            if current_best < best_fitness:
                best_fitness = current_best
                best_individual = current_best_individual
                no_improvement_counter = 0  # Reset counter
            else:
                no_improvement_counter += 1
        elif problem == 'knapsack':
            fitnesses = [knapsack_fitness(ind, knapsack_values, knapsack_weights, knapsack_max_weight) for ind in population]
            current_best = max(fitnesses)
            current_best_individual = population[fitnesses.index(current_best)]
            current_average = sum(fitnesses) / len(fitnesses)
            current_std = np.std(fitnesses)
            if current_best > best_fitness:
                best_fitness = current_best
                best_individual = current_best_individual
                no_improvement_counter = 0  # Reset counter
            else:
                no_improvement_counter += 1

        # Check for convergence
        if no_improvement_counter >= no_improvement_generations:
            generations_to_converge = gen
            break  # Converged

        # Store fitness data
        best_fitness_history.append(best_fitness)
        average_fitness_history.append(current_average)
        std_fitness_history.append(current_std)
        # print(f"Generation {gen}: Best Fitness = {best_fitness}")

        # Selection
        if selection_method == 'tournament':
            selected = tournament_selection(population, fitnesses, problem= problem)
        elif selection_method == 'roulette':
            selected = roulette_wheel_selection(population, fitnesses, problem=problem)
        else:
            raise ValueError("Unsupported selection method.")

        # Crossover
        offspring = []
        for i in range(0, pop_size, 2):
            parent1 = selected[i]
            parent2 = selected[i+1] if i+1 < pop_size else selected[0]
            if random.random() < crossover_rate:
                if crossover_method == 'single_point':
                    child1, child2 = single_point_crossover(parent1, parent2)
                elif crossover_method == 'two_point':
                    child1, child2 = two_point_crossover(parent1, parent2)
                elif crossover_method == 'uniform':
                    child1, child2 = uniform_crossover(parent1, parent2)
                else:
                    raise ValueError("Unsupported crossover method.")
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            offspring.extend([child1, child2])

        # Mutation
        for i in range(pop_size):
            if problem == 'rastrigin':
                if mutation_method == 'gaussian':
                    offspring[i] = gaussian_mutation(offspring[i], mutation_rate)
                else:
                    raise ValueError("Unsupported mutation method for continuous problems.")
            elif problem == 'knapsack':
                if mutation_method == 'flip_bit':
                    offspring[i] = flip_bit_mutation(offspring[i], mutation_rate)
                else:
                    raise ValueError("Unsupported mutation method for discrete problems.")

        # Create new population
        population = offspring

    return best_individual, best_fitness, best_fitness_history, average_fitness_history, std_fitness_history, generations_to_converge

# Function to Plot Fitness History
def plot_fitness_history(best_fitness, average_fitness, std_fitness, problem_name):
    generations = range(len(best_fitness))
    plt.figure(figsize=(12, 6))
    plt.plot(generations, best_fitness, label='Best Fitness')
    plt.plot(generations, average_fitness, label='Average Fitness')
    plt.fill_between(generations,
                     np.array(average_fitness) - np.array(std_fitness),
                     np.array(average_fitness) + np.array(std_fitness),
                     color='gray', alpha=0.2, label='Std Dev')
    plt.title(f'Fitness over Generations for {problem_name}')
    plt.xlabel('Generation')
    plt.ylabel('Fitness value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Running the GA
if __name__ == "__main__":
    # Rastrigin Function Optimization
    print("Running GA on Rastrigin Function...")
    (best_solution_rastrigin,
     best_fitness_rastrigin,
     best_hist_rastrigin,
     avg_hist_rastrigin,
     std_hist_rastrigin,
     gen) = genetic_algorithm(
        problem='rastrigin',
        pop_size=100,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1,
        selection_method='tournament',
        crossover_method='single_point',
        mutation_method='gaussian',
        num_dimensions=10
    )
    print("\nBest Solution for Rastrigin Function:")
    print(best_solution_rastrigin)
    print(f"Fitness (Rastrigin): {best_fitness_rastrigin}")

    # Plot for Rastrigin Function
    plot_fitness_history(best_hist_rastrigin, avg_hist_rastrigin, std_hist_rastrigin, "Rastrigin Function")

    # Knapsack Problem Optimization
    print("\nRunning GA on Knapsack Problem...")
    # Example: 10 items
    knapsack_items = {
        'item1': {'value': 60, 'weight': 10},
        'item2': {'value': 100, 'weight': 20},
        'item3': {'value': 120, 'weight': 30},
        'item4': {'value': 80, 'weight': 15},
        'item5': {'value': 40, 'weight': 5},
        'item6': {'value': 70, 'weight': 25},
        'item7': {'value': 90, 'weight': 35},
        'item8': {'value': 150, 'weight': 40},
        'item9': {'value': 200, 'weight': 50},
        'item10': {'value': 30, 'weight': 10}
    }
    knapsack_max_weight = 100

    (best_solution_knapsack,
     best_fitness_knapsack,
     best_hist_knapsack,
     avg_hist_knapsack,
     std_hist_knapsack,
     gen) = genetic_algorithm(
        problem='knapsack',
        pop_size=100,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.05,
        selection_method='roulette',
        crossover_method='two_point',
        mutation_method='flip_bit',
        knapsack_values=[item['value'] for item in knapsack_items.values()],
        knapsack_weights=[item['weight'] for item in knapsack_items.values()],
        knapsack_max_weight=knapsack_max_weight
    )
    print("\nBest Solution for Knapsack Problem:")
    print(best_solution_knapsack)
    print(f"Total Value: {best_fitness_knapsack}")

    # Plot for Knapsack Problem
    plot_fitness_history(best_hist_knapsack, avg_hist_knapsack, std_hist_knapsack, "Knapsack Problem")

# Task 11. Experiment with Genetic Algorithm (GA) Optimization Hyperparameters
def run_experiments():
    """
    Run experiments by varying GA hyperparameters and record the results.
    Generate plots to analyze the impact of hyperparameters on GA performance.
    """
    # Define hyperparameter ranges
    population_sizes = [50, 100, 200]
    crossover_types = ['single_point', 'two_point', 'uniform']
    mutation_rates = [0.01, 0.05, 0.1]
    selection_methods = ['tournament', 'roulette']

    # Generate all possible combinations of hyperparameters
    parameter_combinations = list(
        itertools.product(population_sizes, crossover_types, mutation_rates, selection_methods))

    # Initialize list to store experiment results
    results = []

    # Define problem type: 'rastrigin' or 'knapsack'
    problem = 'knapsack'  # Change to 'knapsack' to run experiments on Knapsack problem

    # Define parameters for Knapsack problem if selected
    if problem == 'knapsack':
        knapsack_items = {
            'item1': {'value': 60, 'weight': 10},
            'item2': {'value': 100, 'weight': 20},
            'item3': {'value': 120, 'weight': 30},
            'item4': {'value': 80, 'weight': 15},
            'item5': {'value': 40, 'weight': 5},
            'item6': {'value': 70, 'weight': 25},
            'item7': {'value': 90, 'weight': 35},
            'item8': {'value': 150, 'weight': 40},
            'item9': {'value': 200, 'weight': 50},
            'item10': {'value': 30, 'weight': 10}
        }

        knapsack_max_weight = 100

    # Iterate over all hyperparameter combinations
    for idx, (pop_size, crossover, mutation_rate, selection) in enumerate(parameter_combinations):
        print(f"\nRunning Experiment {idx + 1}/{len(parameter_combinations)}")
        print(
            f"Parameters: Population Size={pop_size}, Crossover Type={crossover}, Mutation Rate={mutation_rate}, Selection Method={selection}")

        # Run GA
        if problem == 'rastrigin':
            best_ind, best_fit, best_hist, avg_hist, std_hist, gens = genetic_algorithm(
                problem='rastrigin',
                pop_size=pop_size,
                generations=100,
                crossover_rate=0.8,
                mutation_rate=mutation_rate,
                selection_method=selection,
                crossover_method=crossover,
                mutation_method='gaussian',
                num_dimensions=10,
                no_improvement_generations=10  # Stop if no improvement for 10 generations
            )
        elif problem == 'knapsack':
            best_ind, best_fit, best_hist, avg_hist, std_hist, gens = genetic_algorithm(
                problem='knapsack',
                pop_size=pop_size,
                generations=100,
                crossover_rate=0.8,
                mutation_rate=mutation_rate,
                selection_method=selection,
                crossover_method=crossover,
                mutation_method='flip_bit',
                knapsack_values=[item['value'] for item in knapsack_items.values()],
                knapsack_weights=[item['weight'] for item in knapsack_items.values()],
                knapsack_max_weight=knapsack_max_weight,
                no_improvement_generations=10
            )
        else:
            raise ValueError("Unsupported problem type.")

        # Record results
        results.append({
            'Population Size': pop_size,
            'Crossover Type': crossover,
            'Mutation Rate': mutation_rate,
            'Selection Method': selection,
            'Best Fitness': best_fit,
            'Generations to Converge': gens
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV (optional)
    results_df.to_csv('ga_hyperparameters_experiment_results.csv', index=False)

    print("\nAll experiments completed. Results saved to 'ga_hyperparameters_experiment_results.csv'.")

    # Display results
    print("\nExperiment Results:")
    print(results_df)

    # Analyze results
    analyze_results(results_df, problem)


def analyze_results(results_df, problem_name):
    """
    Analyze and plot the results of the GA hyperparameter experiments.

    Parameters:
        results_df (DataFrame): DataFrame containing experiment results.
        problem_name (str): Name of the problem ('rastrigin' or 'knapsack').
    """
    # Set seaborn style for better aesthetics
    sns.set(style="whitegrid")

    # Plot Best Fitness by Population Size
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Population Size', y='Best Fitness', data=results_df)
    plt.title(f'Best Fitness by Population Size for {problem_name.capitalize()}')
    plt.xlabel('Population Size')
    plt.ylabel('Best Fitness')
    plt.show()

    # Plot Best Fitness by Crossover Type
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Crossover Type', y='Best Fitness', data=results_df)
    plt.title(f'Best Fitness by Crossover Type for {problem_name.capitalize()}')
    plt.xlabel('Crossover Type')
    plt.ylabel('Best Fitness')
    plt.show()

    # Plot Best Fitness by Mutation Rate
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Mutation Rate', y='Best Fitness', data=results_df)
    plt.title(f'Best Fitness by Mutation Rate for {problem_name.capitalize()}')
    plt.xlabel('Mutation Rate')
    plt.ylabel('Best Fitness')
    plt.show()

    # Plot Best Fitness by Selection Method
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Selection Method', y='Best Fitness', data=results_df)
    plt.title(f'Best Fitness by Selection Method for {problem_name.capitalize()}')
    plt.xlabel('Selection Method')
    plt.ylabel('Best Fitness')
    plt.show()

    # Correlation Heatmap (Only Numeric Columns)
    plt.figure(figsize=(10, 8))
    numeric_df = results_df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap of Hyperparameters and Performance Metrics')
        plt.show()
    else:
        print("No numeric columns available for correlation heatmap.")

    # Identify Hyperparameter Combinations that Prevent Convergence
    # Define a threshold for "good" fitness (adjust based on problem)
    if problem_name == 'rastrigin':
        good_fitness_threshold = 1.0  # Close to global minimum (0)
    elif problem_name == 'knapsack':
        # Assuming maximum possible fitness is sum of all values
        if 'knapsack_values' in globals():
            good_fitness_threshold = sum([item['value'] for item in knapsack_items.values()]) * 0.9  # 90% of max
        else:
            good_fitness_threshold = None
    else:
        good_fitness_threshold = None

    if good_fitness_threshold is not None:
        non_converged = results_df[
            (results_df['Generations to Converge'] == results_df['Generations to Converge'].max()) &
            (results_df['Best Fitness'] > good_fitness_threshold)
            ]

        if not non_converged.empty:
            print("\nHyperparameter Combinations that Likely Did Not Converge:")
            print(non_converged)

            # Plot these combinations
            plt.figure(figsize=(12, 6))
            sns.scatterplot(
                data=non_converged,
                x='Mutation Rate',
                y='Best Fitness',
                hue='Selection Method',
                style='Crossover Type',
                s=100
            )
            plt.title(f'Non-Converged Hyperparameter Combinations for {problem_name.capitalize()}')
            plt.xlabel('Mutation Rate')
            plt.ylabel('Best Fitness')
            plt.legend()
            plt.show()
        else:
            print("\nAll hyperparameter combinations converged successfully.")
    else:
        print("\nGood fitness threshold not defined for the problem.")

if __name__ == "__main__":
    run_experiments()
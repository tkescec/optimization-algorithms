import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools
import pandas as pd
from tqdm import tqdm


# Task 8. Implement Ant Colony Optimization (ACO) for the Traveling Salesman Problem (TSP)

class AntColonyOptimizer:
    def __init__(self, distance_matrix, coordinates, num_ants, num_iterations, alpha=1.0, beta=5.0,
                 evaporation_rate=0.5, pheromone_deposit=100, start_city=None):
        """
        Initializes the ACO algorithm parameters.

        :param distance_matrix: 2D numpy array representing distances between cities.
        :param coordinates: List of tuples representing (x, y) coordinates of cities.
        :param num_ants: Number of ants in the colony.
        :param num_iterations: Number of iterations to run the algorithm.
        :param alpha: Influence of pheromone on direction.
        :param beta: Influence of heuristic information (inverse of distance).
        :param evaporation_rate: Rate at which pheromone evaporates.
        :param pheromone_deposit: Constant for pheromone deposition.
        :param start_city: Optional index of the starting city for all ants.
        """
        self.distance_matrix = distance_matrix
        self.coordinates = coordinates
        self.num_cities = distance_matrix.shape[0]
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        self.start_city = start_city

        # Initialize pheromone levels to a small constant
        self.pheromone_matrix = np.ones((self.num_cities, self.num_cities)) * 0.1

        # Heuristic information: inverse of distance, with handling for zero distance
        with np.errstate(divide='ignore'):
            self.heuristic_matrix = 1 / self.distance_matrix
            self.heuristic_matrix[self.distance_matrix == 0] = 0

        # Keep track of the best solution
        self.best_tour = None
        self.best_distance = float('inf')

        # For visualization purposes
        self.all_best_tours = []
        self.all_best_distances = []

    def run(self):
        """
        Executes the ACO algorithm.

        :return: Best tour and its total distance.
        """
        for iteration in range(self.num_iterations):
            all_tours = []
            all_distances = []

            for ant in range(self.num_ants):
                tour = self.construct_solution()
                distance = self.calculate_total_distance(tour)
                all_tours.append(tour)
                all_distances.append(distance)

                # Update best tour if found
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_tour = tour

            self.update_pheromones(all_tours, all_distances)

            # Store best tour and distance for each iteration
            self.all_best_tours.append(self.best_tour)
            self.all_best_distances.append(self.best_distance)

            # print(f"Iteration {iteration + 1}/{self.num_iterations}, Best Distance: {self.best_distance:.2f}")

        return self.best_tour, self.best_distance

    def construct_solution(self):
        """
        Constructs a solution (tour) for a single ant.

        :return: List representing the tour.
        """
        tour = []
        if self.start_city is not None:
            current_city = self.start_city
        else:
            # Start from a random city
            current_city = random.randint(0, self.num_cities - 1)
        tour.append(current_city)
        visited = set(tour)

        while len(tour) < self.num_cities:
            probabilities = self.calculate_transition_probabilities(current_city, visited)
            next_city = self.select_next_city(probabilities)
            tour.append(next_city)
            visited.add(next_city)
            current_city = next_city

        # Return to the starting city
        tour.append(tour[0])
        return tour

    def calculate_transition_probabilities(self, current_city, visited):
        """
        Calculates the probability of moving to each city.

        :param current_city: Current city of the ant.
        :param visited: Set of already visited cities.
        :return: Probability distribution over next cities.
        """
        pheromone = np.copy(self.pheromone_matrix[current_city])
        heuristic = np.copy(self.heuristic_matrix[current_city])

        # Apply alpha and beta
        pheromone = np.power(pheromone, self.alpha)
        heuristic = np.power(heuristic, self.beta)

        # Set pheromone and heuristic to zero for visited cities
        pheromone[list(visited)] = 0
        heuristic[list(visited)] = 0

        # Calculate probabilities
        probabilities = pheromone * heuristic
        total = probabilities.sum()
        if total == 0:
            # If all probabilities are zero (due to visited cities), choose randomly among unvisited
            probabilities = np.array([1 if i not in visited else 0 for i in range(self.num_cities)])
            probabilities = probabilities / probabilities.sum()
        else:
            probabilities = probabilities / total

        return probabilities

    def select_next_city(self, probabilities):
        """
        Selects the next city based on the probability distribution.

        :param probabilities: Probability distribution over next cities.
        :return: Index of the selected city.
        """
        return np.random.choice(range(self.num_cities), p=probabilities)

    def calculate_total_distance(self, tour):
        """
        Calculates the total distance of the tour.

        :param tour: List representing the tour.
        :return: Total distance of the tour.
        """
        distance = 0
        for i in range(len(tour) - 1):
            distance += self.distance_matrix[tour[i], tour[i + 1]]
        return distance

    def update_pheromones(self, all_tours, all_distances):
        """
        Updates the pheromone levels on all edges.

        :param all_tours: List of all tours constructed in this iteration.
        :param all_distances: List of total distances for each tour.
        """
        # Evaporation
        self.pheromone_matrix *= (1 - self.evaporation_rate)

        # Deposit new pheromones
        for tour, distance in zip(all_tours, all_distances):
            pheromone_contribution = self.pheromone_deposit / distance
            for i in range(len(tour) - 1):
                from_city = tour[i]
                to_city = tour[i + 1]
                self.pheromone_matrix[from_city][to_city] += pheromone_contribution
                self.pheromone_matrix[to_city][from_city] += pheromone_contribution  # Assuming undirected graph

    def get_pheromone_matrix(self):
        """
        Returns the current pheromone matrix.

        :return: Pheromone matrix.
        """
        return self.pheromone_matrix


def plot_tour(tour, coordinates, cities, pheromone_matrix, distance_matrix):
    """
    Displays the best tour on a graph with unique colors for each route,
    and annotates distances and pheromone levels.

    :param tour: List representing the best tour.
    :param coordinates: List of tuples representing (x, y) coordinates of cities.
    :param cities: List of city names.
    :param pheromone_matrix: Current pheromone matrix.
    :param distance_matrix: Current distance matrix.
    """
    plt.figure(figsize=(14, 10))

    # Plot cities
    x = [coord[0] for coord in coordinates]
    y = [coord[1] for coord in coordinates]
    plt.scatter(x, y, c='red', zorder=5, s=100)

    for i, city in enumerate(cities):
        plt.text(x[i] + 0.5, y[i] + 0.5, city, fontsize=12, ha='right')

    # Define a color map
    colors = list(mcolors.TABLEAU_COLORS.values())
    if len(tour) - 1 > len(colors):
        # Extend the color list if there are more edges than colors
        colors = colors * ((len(tour) - 1) // len(colors) + 1)

    # Plot each edge with a unique color
    for i in range(len(tour) - 1):
        from_city = tour[i]
        to_city = tour[i + 1]
        plt.plot([coordinates[from_city][0], coordinates[to_city][0]],
                 [coordinates[from_city][1], coordinates[to_city][1]],
                 color=colors[i], linewidth=2, label=f'Route {i + 1}' if i == 0 else "")

        # Calculate midpoint for annotation
        mid_x = (coordinates[from_city][0] + coordinates[to_city][0]) / 2
        mid_y = (coordinates[from_city][1] + coordinates[to_city][1]) / 2

        # Get distance and pheromone level
        distance = distance_matrix[from_city][to_city]
        pheromone = pheromone_matrix[from_city][to_city]

        # Annotate distance and pheromone
        plt.text(mid_x, mid_y, f"{distance:.2f}\n{pheromone:.2f}", fontsize=8,
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.title('Best Tour with Distances and Pheromone Levels', fontsize=16)
    plt.xlabel('X Coordinate', fontsize=14)
    plt.ylabel('Y Coordinate', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Define 10 cities with (x, y) coordinates
    cities = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    coordinates = [
        (0, 0),  # City A
        (20, 10),  # City B
        (10, 20),  # City C
        (30, 30),  # City D
        (40, 10),  # City E
        (50, 20),  # City F
        (60, 30),  # City G
        (70, 10),  # City H
        (80, 20),  # City I
        (90, 30)  # City J
    ]

    # Calculate distance matrix using Euclidean distance
    num_cities = len(cities)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance_matrix[i][j] = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
            else:
                distance_matrix[i][j] = 0.0

    # Parameters
    num_ants = 20
    num_iterations = 100
    alpha = 1  # Influence of pheromone
    beta = 5  # Influence of heuristic (inverse distance)
    evaporation_rate = 0.5
    pheromone_deposit = 100
    start_city = 0  # Index of the starting city (e.g., 0 for City A)

    # Initialize ACO
    aco = AntColonyOptimizer(
        distance_matrix=distance_matrix,
        coordinates=coordinates,
        num_ants=num_ants,
        num_iterations=num_iterations,
        alpha=alpha,
        beta=beta,
        evaporation_rate=evaporation_rate,
        pheromone_deposit=pheromone_deposit,
        start_city=start_city
    )

    # Run ACO
    best_tour, best_distance = aco.run()

    # Convert tour indices to city names
    best_tour_cities = [cities[i] for i in best_tour]

    print("\nBest Tour Found:")
    print(" -> ".join(best_tour_cities))
    print(f"Total Distance: {best_distance:.2f}")

    # Plot the best tour
    pheromone_matrix = aco.get_pheromone_matrix()
    plot_tour(best_tour, coordinates, cities, pheromone_matrix, distance_matrix)

if __name__ == "__main__":
    main()

# Task 9. Experiment with Ant Colony Optimization (ACO) for the Traveling Salesman Problem (TSP)
def plot_results(results_df, hyperparameter, ylabel, title):
    """
    Plots the average best distance against a hyperparameter.

    :param results_df: DataFrame containing the results.
    :param hyperparameter: The hyperparameter to plot on the x-axis.
    :param ylabel: Label for the y-axis.
    :param title: Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    grouped = results_df.groupby(hyperparameter)['Best Distance'].mean().reset_index()
    plt.plot(grouped[hyperparameter], grouped['Best Distance'], marker='o')
    plt.xlabel(hyperparameter)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"plots/{title}.png")
    plt.show()


def run_experiments():
    """
    Runs ACO experiments with different hyperparameter combinations and records the results.
    """
    # Define 10 cities with (x, y) coordinates
    cities = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    coordinates = [
        (0, 0),  # City A
        (20, 10),  # City B
        (10, 20),  # City C
        (30, 30),  # City D
        (40, 10),  # City E
        (50, 20),  # City F
        (60, 30),  # City G
        (70, 10),  # City H
        (80, 20),  # City I
        (90, 30)  # City J
    ]

    # Calculate distance matrix using Euclidean distance
    num_cities = len(cities)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance_matrix[i][j] = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
            else:
                distance_matrix[i][j] = 0.0

    # Define hyperparameter ranges
    alpha_values = [0.5, 1.0, 2.0]
    beta_values = [2.0, 5.0, 10.0]
    evaporation_rates = [0.3, 0.5, 0.7]
    num_ants_values = [10, 20, 30]

    # Number of trials per combination
    num_trials = 30

    # Initialize a list to store all results
    results = []

    # Define starting city (e.g., 0 for City A)
    start_city = 0

    # Iterate over all combinations of hyperparameters
    hyperparameter_combinations = list(itertools.product(alpha_values, beta_values, evaporation_rates, num_ants_values))

    print(f"Total combinations: {len(hyperparameter_combinations)}")
    print(f"Total trials: {len(hyperparameter_combinations) * num_trials}")

    for alpha, beta, rho, num_ants in tqdm(hyperparameter_combinations, desc="Hyperparameter Combinations"):
        for trial in range(num_trials):
            # Initialize ACO
            aco = AntColonyOptimizer(
                distance_matrix=distance_matrix,
                coordinates=coordinates,
                num_ants=num_ants,
                num_iterations=100,  # Keeping iterations constant
                alpha=alpha,
                beta=beta,
                evaporation_rate=rho,
                pheromone_deposit=100,
                start_city=start_city
            )

            # Run ACO
            best_tour, best_distance = aco.run()

            # Record the results
            results.append({
                'Alpha': alpha,
                'Beta': beta,
                'Evaporation Rate': rho,
                'Number of Ants': num_ants,
                'Trial': trial + 1,
                'Best Distance': best_distance
            })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    # results_df.to_csv('aco_experiment_results.csv', index=False)
    # print("Experiments completed and results saved to 'aco_experiment_results.csv'.")

    return results_df


def main():
    """
    Main function to run experiments and visualize results.
    """
    # Run experiments
    results_df = run_experiments()

    # Visualization: Plot average best distance vs each hyperparameter

    # 1. Alpha vs Best Distance
    plot_results(
        results_df,
        hyperparameter='Alpha',
        ylabel='Average Best Distance',
        title='Effect of Alpha on ACO Performance'
    )

    # 2. Beta vs Best Distance
    plot_results(
        results_df,
        hyperparameter='Beta',
        ylabel='Average Best Distance',
        title='Effect of Beta on ACO Performance'
    )

    # 3. Evaporation Rate vs Best Distance
    plot_results(
        results_df,
        hyperparameter='Evaporation Rate',
        ylabel='Average Best Distance',
        title='Effect of Evaporation Rate on ACO Performance'
    )

    # 4. Number of Ants vs Best Distance
    plot_results(
        results_df,
        hyperparameter='Number of Ants',
        ylabel='Average Best Distance',
        title='Effect of Number of Ants on ACO Performance'
    )

    # Save the results DataFrame to a CSV file for further analysis
    results_df.to_csv('aco_experiment_results.csv', index=False)
    print("All plots generated and results saved to 'aco_experiment_results.csv'.")


if __name__ == "__main__":
    main()
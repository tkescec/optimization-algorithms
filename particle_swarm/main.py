from tkinter.font import names

import numpy as np
import random
import matplotlib.pyplot as plt
import time
from itertools import product


# Task 5. Implement Particle Swarm Optimization algorithm

# Define the objective function (Rosenbrock function)
def rosenbrock_function(position):
    """
    Compute the Rosenbrock function.
    f(x) = sum_{i=1}^{n-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    Global minimum is at (1,1,...,1) where f(x)=0.
    """
    return sum(100.0 * (position[1:] - position[:-1] ** 2.0) ** 2.0 + (1 - position[:-1]) ** 2.0)


# Define the Particle class
class Particle:
    def __init__(self, dimension, bounds):
        """
        Initialize a particle with random position and velocity.
        :param dimension: Number of dimensions
        :param bounds: Tuple (min, max) for each dimension
        """
        self.position = np.random.uniform(bounds[0], bounds[1], dimension)
        self.velocity = np.random.uniform(-abs(bounds[1] - bounds[0]), abs(bounds[1] - bounds[0]), dimension)
        self.best_position = np.copy(self.position)
        self.best_value = rosenbrock_function(self.position)

    def update_velocity(self, global_best_position, inertia, C1, C2):
        """
        Update the velocity of the particle.
        :param global_best_position: The best position found by the swarm
        :param inertia: Inertia coefficient
        :param C1: Cognitive coefficient
        :param C2: Social coefficient
        """
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))

        cognitive_velocity = C1 * r1 * (self.best_position - self.position)
        social_velocity = C2 * r2 * (global_best_position - self.position)
        self.velocity = inertia * self.velocity + cognitive_velocity + social_velocity

    def update_position_func(self, bounds):
        """
        Update the position of the particle based on its velocity.
        :param bounds: Tuple (min, max) for each dimension
        """
        self.position += self.velocity
        # Ensure the particle stays within bounds
        self.position = np.clip(self.position, bounds[0], bounds[1])

        # Update personal best if necessary
        current_value = rosenbrock_function(self.position)
        if current_value < self.best_value:
            self.best_value = current_value
            self.best_position = np.copy(self.position)


# Define the Swarm class
class Swarm:
    def __init__(self, num_particles, dimension, bounds):
        """
        Initialize the swarm with a number of particles.
        :param num_particles: Number of particles in the swarm
        :param dimension: Number of dimensions
        :param bounds: Tuple (min, max) for each dimension
        """
        self.particles = [Particle(dimension, bounds) for _ in range(num_particles)]
        # Initialize global best
        self.global_best_particle = min(self.particles, key=lambda p: p.best_value)
        self.global_best_position = np.copy(self.global_best_particle.best_position)
        self.global_best_value = self.global_best_particle.best_value
        # To track best fitness over iterations
        self.best_fitness_history = [self.global_best_value]

    def run_optimization(self, max_iterations, inertia, C1, C2, bounds, convergence_threshold=1e-6):
        """
        Execute the optimization process.
        :param max_iterations: Maximum number of iterations
        :param inertia: Inertia coefficient
        :param C1: Cognitive coefficient
        :param C2: Social coefficient
        :param bounds: Tuple (min, max) for each dimension
        :return: Best position and its fitness value
        """
        iterations_taken = max_iterations
        converged = False  # Flag to indicate convergence
        for iteration in range(max_iterations):
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, inertia, C1, C2)
                particle.update_position_func(bounds)

                # Update global best if necessary
                if particle.best_value < self.global_best_value:
                    self.global_best_value = particle.best_value
                    self.global_best_position = np.copy(particle.best_position)

            self.best_fitness_history.append(self.global_best_value)

            # Optional: Print progress
            # if (iteration + 1) % max(1, max_iterations // 10) == 0 or iteration == 0:
            #     print(f"Iteration {iteration + 1}/{max_iterations}, Best Fitness: {self.global_best_value:.6f}")

            # Check for convergence
            if self.global_best_value < convergence_threshold:
                iterations_taken = iteration + 1
                converged = True
                print(f"Converged at iteration {iterations_taken} with best fitness {self.global_best_value:.6f}")
                break

        return self.global_best_position, self.global_best_value, iterations_taken, converged

    def get_best_position(self):
        """
        Return the best position found by the swarm.
        """
        return self.global_best_position

    def get_best_value(self):
        """
        Return the best fitness value found by the swarm.
        """
        return self.global_best_value

    def get_fitness_history(self):
        """
        Return the history of best fitness values over iterations.
        """
        return self.best_fitness_history


# Function to plot the swarm (supports 2D )
def plot_swarm(swarm, bounds, name=None):
    plt.figure(figsize=(10, 10))
    # Plot all particles
    for particle in swarm.particles:
        plt.plot(particle.position[0], particle.position[1], 'ro', markersize=4)
    # Plot the best position
    plt.plot(swarm.get_best_position()[0], swarm.get_best_position()[1], 'bo', markersize=8, label='Best Position')
    plt.xlim(bounds[0], bounds[1])
    plt.ylim(bounds[0], bounds[1])
    plt.grid(True)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Particle Swarm Optimization (2D)')
    plt.legend()

    if name:
        plt.savefig(f"plots/{name}.png")

    plt.show()


# Function to plot the convergence of best fitness over iterations
def plot_convergence(swarm, name=None):
    plt.figure(figsize=(10, 6))
    iterations = range(len(swarm.get_fitness_history()))
    plt.plot(iterations, swarm.get_fitness_history(), 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.title('Convergence of Best Fitness Over Iterations')
    plt.legend([f"Config: {name}"])
    plt.grid(True)
    plt.yscale('log')  # Log scale for better visibility
    if name:
        plt.savefig(f"plots/{name}_convergence.png")
    plt.show()


# Main function
def main():
    # Define parameters
    dimension = 2  # Number of dimensions (e.g., 2, 3, 5, etc.)
    num_particles = 100  # Number of particles in the swarm
    bounds = (-10, 10)  # Search space boundaries for each dimension
    max_iterations = 200  # Maximum number of iterations
    inertia = 0.729  # Inertia weight to prevent velocities from becoming too large
    C1 = 1.49445 # Cognitive (particle) coefficient
    C2 = 1.49445 # Social (swarm) coefficient

    # Initialize the swarm
    swarm = Swarm(num_particles, dimension, bounds)

    # Run the optimization
    best_position, best_value, iterations_taken, converged = swarm.run_optimization(
        max_iterations, inertia, C1, C2, bounds
    )

    # Plot the swarm positions if 2D or 3D
    plot_swarm(swarm, bounds)
    # if dimension == 2:
    #     plot_swarm(swarm, bounds)
    # else:
    #     print("Skipping swarm position plot for dimensions higher than 3.")

    # Plot the convergence graph
    plot_convergence(swarm)

    # Display the best result
    print("\nOptimization Completed.")
    print(f"Best Position: {best_position}")
    print(f"Best Fitness: {best_value:.6f}")


if __name__ == "__main__":
    main()


# Task 6. Experiment with C1 and C2 hyperparameters

# Function to pad fitness histories
def pad_fitness_histories(fitness_histories, max_length):
    """
    Pad each fitness history list to the max_length by appending the last fitness value.

    :param fitness_histories: List of lists containing fitness values per run.
    :param max_length: The length to pad each fitness history to.
    :return: A new list with padded fitness histories.
    """
    padded_histories = []
    for history in fitness_histories:
        if len(history) < max_length:
            # Extend the history by repeating the last value
            padded_history = history + [history[-1]] * (max_length - len(history))
        else:
            padded_history = history
        padded_histories.append(padded_history)
    return padded_histories


# Function to plot the convergence curve
def plot_convergence_curve(plot_configs, config_labels, max_iterations):
    """
    Plot the convergence curves for different PSO configurations.

    :param plot_configs: List of result dictionaries for selected configurations.
    :param config_labels: List of labels for each configuration.
    :param max_iterations: Maximum number of iterations to plot.
    """
    plt.figure(figsize=(12, 8))

    for config, label in zip(plot_configs, config_labels):
        fitness_histories = config['fitness_histories']
        # Determine the maximum length of fitness histories
        max_length = max(len(history) for history in fitness_histories)
        # Pad all histories to the maximum length
        padded_histories = pad_fitness_histories(fitness_histories, max_length)
        # Compute the mean fitness at each iteration
        mean_fitness = np.mean(padded_histories, axis=0)
        # Truncate or pad the mean_fitness to max_iterations
        if len(mean_fitness) > max_iterations:
            mean_fitness = mean_fitness[:max_iterations]
        elif len(mean_fitness) < max_iterations:
            # Use np.full to create an array of the last fitness value
            padding = np.full(max_iterations - len(mean_fitness), mean_fitness[-1])
            mean_fitness = np.concatenate([mean_fitness, padding])
        # Plot the convergence curve
        plt.plot(mean_fitness, label=label)

    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.title('Convergence Curves for Different PSO Configurations')
    plt.yscale('log')  # Log scale for better visibility
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig("plots/convergence_curves.png")
    plt.show()


# Function to plot the distribution of best solutions
def plot_best_solutions_distribution(results):
    """
    Plot the distribution of best solutions for different configurations.

    :param results: List of result dictionaries for all configurations.
    """
    # Extract average best solutions
    avg_best_solutions = [config['avg_best_solution'] for config in results]
    config_labels = [f"c1={config['c1']}, c2={config['c2']}, w={config['w']}" for config in results]

    plt.figure(figsize=(20, 10))
    plt.bar(range(len(avg_best_solutions)), avg_best_solutions, tick_label=config_labels)
    plt.xlabel('Configuration')
    plt.ylabel('Average Best Solution')
    plt.title('Average Best Solutions Across Configurations')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("plots/best_solutions_distribution.png")
    plt.show()

# Function to perform experiments
def conduct_experiments():
    # Define hyperparameter ranges
    c1_values = [0.5, 1.0, 1.5, 2.0]
    c2_values = [0.5, 1.0, 1.5, 2.0]
    w_values = [0.4, 0.6, 0.8, 1.0]

    # Generate all combinations
    hyperparameter_combinations = list(product(c1_values, c2_values, w_values))

    # Experimental parameters
    num_runs = 30
    num_particles = 100
    dimension = 2  # For visualization purposes
    bounds = (-10, 10)
    max_iterations = 200
    convergence_threshold = 1e-6

    # Data storage
    results = []

    # Iterate through each hyperparameter combination
    for idx, (c1, c2, w) in enumerate(hyperparameter_combinations):
        print(f"\nRunning Configuration {idx + 1}/{len(hyperparameter_combinations)}: c1={c1}, c2={c2}, w={w}")
        best_solutions = []
        iterations_to_converge = []
        run_times = []
        fitness_histories = []
        convergence_flags = []  # To track convergence status

        for run in range(num_runs):
            swarm = Swarm(num_particles, dimension, bounds)
            start_time = time.time()
            best_pos, best_val, iterations, converged  = swarm.run_optimization(
                max_iterations, w, c1, c2, bounds, convergence_threshold
            )
            end_time = time.time()

            # Optional: Plot the swarm and convergence curve for the first run
            name = f"config_{idx + 1}_run_{run + 1}_c1_{c1}_c2_{c2}_w_{w}"
            if run == 0:
                plot_swarm(swarm, bounds, name)
                plot_convergence(swarm, name)

            best_solutions.append(best_val)
            iterations_to_converge.append(iterations)
            run_times.append(end_time - start_time)
            fitness_histories.append(swarm.get_fitness_history())
            convergence_flags.append(converged)

        # Compute average metrics
        avg_best_solution = np.mean(best_solutions)
        std_best_solution = np.std(best_solutions)
        avg_iterations = np.mean(iterations_to_converge)
        std_iterations = np.std(iterations_to_converge)
        avg_time = np.mean(run_times)
        std_time = np.std(run_times)
        convergence_rate = sum(convergence_flags) / num_runs  # Percentage of runs that converged

        # Store the results
        results.append({
            'c1': c1,
            'c2': c2,
            'w': w,
            'avg_best_solution': avg_best_solution,
            'std_best_solution': std_best_solution,
            'avg_iterations': avg_iterations,
            'std_iterations': std_iterations,
            'avg_time': avg_time,
            'std_time': std_time,
            'fitness_histories': fitness_histories,
            'convergence_rate': convergence_rate
        })

        print(f"Average Best Solution: {avg_best_solution:.6f} ± {std_best_solution:.6f}")
        print(f"Average Iterations to Converge: {avg_iterations:.2f} ± {std_iterations:.2f}")
        print(f"Average Time per Run: {avg_time:.4f} seconds ± {std_time:.4f} seconds")
        print(f"Convergence Rate: {convergence_rate * 100:.2f}%")

    # Identify configurations that rarely or never converged
    # For example, configurations with convergence rate below 50%
    threshold_rate = 0.5  # 50%
    non_converged_configs = [config for config in results if config['convergence_rate'] < threshold_rate]

    # Plotting convergence curves for selected configurations
    # For demonstration, plot convergence curves for the first 5 configurations
    # You can adjust this as needed or select specific configurations based on results
    plot_configs = results[:5]  # Adjust as needed
    config_labels = [f"c1={c['c1']}, c2={c['c2']}, w={c['w']}" for c in plot_configs]
    plot_convergence_curve(plot_configs, config_labels, max_iterations)

    # Plotting distribution of best solutions
    plot_best_solutions_distribution(results)

    # Display configurations that did not converge
    if non_converged_configs:
        print("\nConfigurations that likely did not converge (Convergence Rate < 50%):")
        for config in non_converged_configs:
            print(f"c1={config['c1']}, c2={config['c2']}, w={config['w']}, "
                  f"Avg Iterations={config['avg_iterations']}, "
                  f"Convergence Rate={config['convergence_rate'] * 100:.2f}%")
    else:
        print("\nAll configurations converged in at least 50% of the runs.")

    return results


def analyze_convergence(results):
    # Find the configuration with c1=0.5, c2=2.0, w=1.0
    target_config = next(
        config for config in results if config['c1'] == 0.5 and config['c2'] == 2.0 and config['w'] == 1.0)

    # Extract relevant data
    fitness_histories = target_config['fitness_histories']
    convergence_flags = target_config['convergence_flags']

    # Plot the convergence curves for the target configuration
    plt.figure(figsize=(12, 6))
    for idx, history in enumerate(fitness_histories):
        plt.plot(history, alpha=0.3, label=f'Run {idx + 1}' if idx < 5 else "", linewidth=0.8)

    plt.xlabel("Iterations")
    plt.ylabel("Best Fitness")
    plt.title("Trend of the best solution during iterations for c1=0.5, c2=2.0, w=1.0")
    plt.show()

    # Compute average best solution for converged runs
    converged_best_solutions = [best for best, flag in zip(target_config['avg_best_solution'], convergence_flags) if
                                flag]
    if converged_best_solutions:
        avg_best_solution_converged = np.mean(converged_best_solutions)
        std_best_solution_converged = np.std(converged_best_solutions)
        print(
            f"Average best solution for converged runs: {avg_best_solution_converged:.6f} ± {std_best_solution_converged:.6f}")
    else:
        print("No converged runs found for the target configuration.")

    # Compute average best solution for all runs
    print(f"Standard deviation for the best solution: {target_config['std_best_solution']:.6f}")

# if __name__ == "__main__":
#     results = conduct_experiments()
#     analyze_convergence(results)

# Task 7. Experiment with C1=0 and C2=1

def main():
    # Define parameters
    dimension = 2  # Number of dimensions (e.g., 2, 3, 5, etc.)
    num_particles = 100  # Number of particles in the swarm
    bounds = (-10, 10)  # Search space boundaries for each dimension
    max_iterations = 200  # Maximum number of iterations
    inertia = 0.729  # Inertia weight to prevent velocities from becoming too large
    C1 = 0 # Cognitive (particle) coefficient
    C2 = 1 # Social (swarm) coefficient

    # Initialize the swarm
    swarm = Swarm(num_particles, dimension, bounds)

    # Run the optimization
    best_position, best_value, iterations_taken, converged = swarm.run_optimization(
        max_iterations, inertia, C1, C2, bounds
    )

    # Plot the swarm positions if 2D or 3D
    plot_swarm(swarm, bounds)
    # if dimension == 2:
    #     plot_swarm(swarm, bounds)
    # else:
    #     print("Skipping swarm position plot for dimensions higher than 3.")

    # Plot the convergence graph
    plot_convergence(swarm)

    # Display the best result
    print("\nOptimization Completed.")
    print(f"Best Position: {best_position}")
    print(f"Best Fitness: {best_value:.6f}")


if __name__ == "__main__":
    main()


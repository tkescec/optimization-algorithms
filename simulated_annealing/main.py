import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# TASK 3. Implement the simulated annealing algorithm.
def rastrigin(x):
    """
    Rastrigin function.
    Global minimum at x=0 with value 0.
    """
    A = 10
    x = np.array(x)  # Ensures compatibility with numpy arrays
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * math.pi * x), axis=0)


def linear_cooling(T0, alpha, k):
    """
    Linear Cooling Schedule: Tk = T0 - alpha * k
    """
    return T0 - alpha * k


def logarithmic_cooling(T0, k):
    """
    Logarithmic Cooling Schedule: Tk = T0 / log(k + 1)
    """
    if k == 0:
        return T0
    return T0 / math.log(k + 1)


def exponential_cooling(T0, beta, k):
    """
    Exponential Cooling Schedule: Tk = T0 * beta^k
    """
    return T0 * (beta ** k)


def adaptive_cooling(T0, k, acceptance_rate, target_rate=0.4, delta=0.05):
    """
    Adaptive Cooling Schedule:
    Adjusts temperature based on acceptance rate.
    - If acceptance rate > target_rate + delta: Slow down cooling (increase temperature)
    - If acceptance rate < target_rate - delta: Speed up cooling (decrease temperature)
    - Else: Maintain current temperature
    """
    if k == 0:
        return T0
    # Simple adaptive mechanism: adjust based on previous acceptance
    # For illustration, we'll assume acceptance_rate affects the temperature
    # This is a placeholder; in practice, you'd maintain state across iterations
    adjustment = 0
    if acceptance_rate > (target_rate + delta):
        adjustment = 0.95  # Slow down cooling
    elif acceptance_rate < (target_rate - delta):
        adjustment = 1.05  # Speed up cooling
    else:
        adjustment = 1.0  # No change
    return T0 * adjustment


def custom_cooling(T0, k):
    """
    Custom Cooling Schedule:
    Tk = T0 / (1 + sin(k))
    Idea: Incorporate periodic fluctuations to allow occasional exploration.
    """
    return T0 / (1 + math.sin(k))


def simulated_annealing(
        objective_func,
        initial_solution,
        T0,
        cooling_schedule,
        max_iter=1000,
        alpha=0.01,  # For linear cooling
        beta=0.99,  # For exponential cooling
        target_rate=0.4,  # For adaptive cooling
        delta=0.05,  # For adaptive cooling
        custom_params=None,
        tolerance=1e-2
):
    """
    Simulated Annealing Algorithm.

    Parameters:
    - objective_func: The function to minimize.
    - initial_solution: Starting point (list of variables).
    - T0: Initial temperature.
    - cooling_schedule: Function to compute temperature.
    - max_iter: Maximum number of iterations.
    - alpha: Parameter for linear cooling.
    - beta: Parameter for exponential cooling.
    - target_rate, delta: Parameters for adaptive cooling.
    - custom_params: Additional parameters for custom cooling.
    - tolerance: Objective value threshold for convergence.

    Returns:
    - best_solution: Best found solution.
    - best_obj: Objective value of the best solution.
    - history: List of objective values over iterations.
    - trajectory: List of solutions over iterations.
    """
    current_solution = initial_solution[:]
    current_obj = objective_func(current_solution)
    best_solution = current_solution[:]
    best_obj = current_obj
    history = [best_obj]
    trajectory = [current_solution[:]]
    worse_acceptances = 0
    total_worse = 0
    iterations_to_tolerance = None

    for k in range(1, max_iter + 1):
        # Generate neighbor: for simplicity, add a small random value to one dimension
        neighbor = current_solution[:]
        i = random.randint(0, len(neighbor) - 1)
        neighbor[i] += random.uniform(-0.5, 0.5)
        # Ensure neighbor stays within the bounds of the search space
        neighbor[i] = max(min(neighbor[i], 5.12), -5.12)  # Adjust bounds as per the Rastrigin function

        neighbor_obj = objective_func(neighbor)

        # Calculate change in objective
        delta_obj = neighbor_obj - current_obj

        # Decide whether to accept the neighbor
        if delta_obj < 0:
            accept = True
            worse_acceptances += 1  # Not a worse solution, but count as accepted for acceptance rate
        else:
            acceptance_prob = math.exp(-delta_obj / T0) if T0 > 0 else 0
            if random.random() < acceptance_prob:
                accept = True
                worse_acceptances += 1
            else:
                accept = False
            total_worse += 1 if accept else 0

        # Check for convergence
        if accept:
            current_solution = neighbor
            current_obj = neighbor_obj
            if current_obj < best_obj:
                best_solution = current_solution[:]
                best_obj = current_obj
        if delta_obj < 0 or random.random() < math.exp(-delta_obj / T0):
            current_solution = neighbor
            current_obj = neighbor_obj
            if current_obj < best_obj:
                best_solution = current_solution[:]
                best_obj = current_obj
                # Check for convergence
                if best_obj <= tolerance and iterations_to_tolerance is None:
                    iterations_to_tolerance = k

        # Calculate acceptance rate of worse solutions
        # For simplicity, define it as the ratio of accepted worse solutions to total worse solutions
        # Here, 'worse_acceptances' counts both accepted better and worse solutions
        # We need to differentiate between them
        # Modify counters accordingly
        if delta_obj >= 0:
            if accept:
                worse_acceptances += 1
                total_worse += 1
            else:
                total_worse += 1

        # Determine acceptance rate for adaptive cooling
        acceptance_rate = worse_acceptances / k if total_worse > 0 else 0

        # Update temperature
        if cooling_schedule.__name__ == "linear_cooling":
            T = cooling_schedule(T0, alpha, k)
        elif cooling_schedule.__name__ == "logarithmic_cooling":
            T = cooling_schedule(T0, k)
        elif cooling_schedule.__name__ == "exponential_cooling":
            T = cooling_schedule(T0, beta, k)
        elif cooling_schedule.__name__ == "adaptive_cooling":
            T = cooling_schedule(T0, k, acceptance_rate, target_rate, delta)
        elif cooling_schedule.__name__ == "custom_cooling":
            T = cooling_schedule(T0, k)
        else:
            raise ValueError("Unknown cooling schedule.")

        # Ensure temperature doesn't go below a minimum threshold
        T = max(T, 1e-10)

        # Update T0 for the next iteration
        T0 = T

        history.append(best_obj)
        trajectory.append(current_solution[:])  # Append the current solution to the trajectory


    return best_solution, best_obj, history, trajectory, iterations_to_tolerance, acceptance_rate

# Define initial solution generator
def generate_initial_solution(dim=2, lower=-5.12, upper=5.12):
    return [random.uniform(lower, upper) for _ in range(dim)]

# Visualize the Results
def visualize_results(results):
    x = np.linspace(-5.12, 5.12, 100)
    y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin([X, Y])
    trajectory = results['Trajectory']
    x_trajectory, y_trajectory = trajectory[:, 0], trajectory[:, 1]

    cooling_schedules = results['Cooling Schedule']

    plt.figure(figsize=(10, 7))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
    plt.plot(x_trajectory, y_trajectory, color='blue', marker='o', markersize=4, label='Solution Path')
    plt.scatter(x_trajectory[0], y_trajectory[0], color='red', label='Final Solution')
    plt.title(f"Solution Path in Search Space ({cooling_schedules} Cooling)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="Objective Value")
    plt.legend()
    plt.grid(True)
    plt.show()


# Define cooling schedules
cooling_schedules = {
    'Linear': linear_cooling,
    'Logarithmic': logarithmic_cooling,
    'Exponential': exponential_cooling,
    'Adaptive': adaptive_cooling,
    'Custom': custom_cooling
}

# Hyperparameters for cooling schedules
cooling_params = {
    'Linear': {'alpha': 0.01},
    'Logarithmic': {},
    'Exponential': {'beta': 0.99},
    'Adaptive': {'target_rate': 0.4, 'delta': 0.05},
    'Custom': {}
}

# Evaluation settings
num_runs = 30
max_iter = 1000
T0 = 100
tolerance = 1e-2  # Objective value ≤ 0.01

# Data storage
results = []

for schedule_name, schedule_func in cooling_schedules.items():
    print(f"Evaluating {schedule_name} Cooling Schedule...")
    for run in range(num_runs):
        initial_solution = generate_initial_solution()
        params = cooling_params.get(schedule_name, {})
        best_sol, best_obj, history, trajectory, iters_to_tol, avg_accept_rate = simulated_annealing(
            objective_func=rastrigin,
            initial_solution=initial_solution,
            T0=T0,
            cooling_schedule=schedule_func,
            max_iter=max_iter,
            alpha=params.get('alpha', 0.01),
            beta=params.get('beta', 0.99),
            target_rate=params.get('target_rate', 0.4),
            delta=params.get('delta', 0.05),
            tolerance=tolerance
        )
        results.append({
            'Cooling Schedule': schedule_name,
            'Run': run + 1,
            'Final Objective': best_obj,
            'Iterations to Tolerance': iters_to_tol if iters_to_tol is not None else max_iter,
            'Acceptance Rate of Worse Solutions': avg_accept_rate,
            'Trajectory': np.array(trajectory),
            'History': np.array(history)
        })

# Store the first occurrence of each cooling schedule
unique_results = {}
for result in results:
    schedule_name = result['Cooling Schedule']
    if schedule_name not in unique_results:
        unique_results[schedule_name] = result

# Visualize the results for each unique cooling schedule
for result in unique_results.values():
    visualize_results(result)

# TASK 4. Evaluate implemented temperature schedulers
# Convert results to DataFrame
df_results = pd.DataFrame(results)
pd.set_option('display.max_columns', None)  # Display all columns
print(df_results)

# Statistical Summary
summary = df_results.groupby('Cooling Schedule').agg({
    'Final Objective': ['mean', 'median', 'min', 'max'],
    'Iterations to Tolerance': ['mean', 'median', 'min', 'max'],
    'Acceptance Rate of Worse Solutions': ['mean', 'median', 'min', 'max']
}).reset_index()

# Rename columns for clarity
summary.columns = ['Cooling Schedule',
                   'Final Obj Mean', 'Final Obj Median', 'Final Obj Min', 'Final Obj Max',
                   'Iterations Mean', 'Iterations Median', 'Iterations Min', 'Iterations Max',
                   'Acceptance Rate Mean', 'Acceptance Rate Median', 'Acceptance Rate Min', 'Acceptance Rate Max']

print("\nStatistical Summary:")
print(summary)

# Boxplots for Final Objective
plt.figure(figsize=(12, 6))
sns.boxplot(x='Cooling Schedule', y='Final Objective', data=df_results)
plt.title('Final Objective Value by Cooling Schedule')
plt.ylabel('Final Objective Value')
plt.yscale('log')  # Log scale due to wide range of objective values
plt.show()

# Boxplots for Iterations to Tolerance
plt.figure(figsize=(12, 6))
sns.boxplot(x='Cooling Schedule', y='Iterations to Tolerance', data=df_results)
plt.title('Iterations to Reach Tolerance by Cooling Schedule')
plt.ylabel('Iterations to Tolerance')
plt.show()

# Boxplots for Acceptance Rate of Worse Solutions
plt.figure(figsize=(12, 6))
sns.boxplot(x='Cooling Schedule', y='Acceptance Rate of Worse Solutions', data=df_results)
plt.title('Acceptance Rate of Worse Solutions by Cooling Schedule')
plt.ylabel('Acceptance Rate')
plt.show()
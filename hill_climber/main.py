import time
import random
import matplotlib.pyplot as plt
import pandas as pd

# TASK 1: Implement the Hill-Climber Algorithm
def generate_neighbors(point, step_size):
    """Generate neighbors by perturbing each dimension of the current point."""
    neighbors = []
    for i in range(len(point)):
        # Create two neighbors for each dimension: one by adding and one by subtracting step_size.
        new_point_plus = point.copy()
        new_point_minus = point.copy()
        new_point_plus[i] += step_size
        new_point_minus[i] -= step_size
        neighbors.append(new_point_plus)
        neighbors.append(new_point_minus)
    return neighbors


def hill_climber(objective_func, n=None, initial_point=None, step_size=0.1, max_iterations=1000):
    """
    Generic Hill-Climber algorithm.
    - objective_func: function to maximize, which accepts a list of coordinates.
    - n: dimension of the problem if no initial point is provided.
    - initial_point: optional starting point as a list of coordinates.
    - step_size: the amount to change each coordinate when exploring neighbors.
    - max_iterations: maximum number of iterations to perform.
    """
    # Initialize starting point
    if initial_point is None:
        if n is None:
            raise ValueError("Provide either `n` for dimensions or an `initial_point`.")
        current_point = [random.uniform(-10, 10) for _ in range(n)]
    else:
        current_point = initial_point.copy()

    current_value = objective_func(current_point)

    for _ in range(max_iterations):
        neighbors = generate_neighbors(current_point, step_size)

        for neighbor in neighbors:
            neighbor_value = objective_func(neighbor)
            # Check if we found a better neighbor
            if neighbor_value > current_value:
                current_point = neighbor
                current_value = neighbor_value

    return current_point, current_value


# === Example objective functions ===

def sphere(point):
    """A simple sphere function (minimized at zero). We return its negative for maximization."""
    return -sum(x ** 2 for x in point)


def parabola_1d(point):
    """1D parabola with a maximum at x = 3."""
    x = point[0]
    return -(x - 3) ** 2 + 10


def parabola_2d(point):
    """2D parabola with a maximum at (3, -2)."""
    x, y = point
    return -(x - 3) ** 2 - (y + 2) ** 2 + 20


# === Usage Examples ===

# 1D Optimization
result_1d = hill_climber(parabola_1d, n=1, step_size=0.1, max_iterations=100)
print("1D result:", result_1d)

# 2D Optimization
result_2d = hill_climber(parabola_2d, n=2, step_size=0.1, max_iterations=100)
print("2D result:", result_2d)

# ND Optimization using the sphere function (e.g., 5 dimensions)
result_nd = hill_climber(sphere, n=5, step_size=0.1, max_iterations=100)
print("5D sphere result:", result_nd)


# TASK 2 Evaluate time complexity of the hill climber algorithm.
# We will evaluate the time complexity of the hill climber algorithm by running it with different input sizes and measuring the total execution time.
max_iter_values = [100, 200, 300, 400, 500, 600]
local_search_values = [50, 100, 150, 200, 250, 300]

fixed_n = 50   # Fixed dimensionality for time_max_iter measurement
fixed_m = 100  # Fixed dimensionality for time_local_search measurement

results = []

# For each combination of input sizes, run the hill climber and measure the total time
for m_val, n_val in zip(max_iter_values, local_search_values):
    # Timing for given max_iter (m) with fixed n
    start = time.time()
    hill_climber(sphere, n=fixed_n, step_size=0.1, max_iterations=m_val)
    time_max_iter = time.time() - start

    # Timing for given local_search (n) with fixed m
    start = time.time()
    hill_climber(sphere, n=n_val, step_size=0.1, max_iterations=fixed_m)
    time_local_search = time.time() - start

    # Total time measurement with current m_val and n_val
    start = time.time()
    hill_climber(sphere, n=n_val, step_size=0.1, max_iterations=m_val)
    total_time = time.time() - start

    # Store the results
    results.append({
        'input_size_max_iter': m_val,
        'input_size_local_search': n_val,
        'time_max_iter (s)': time_max_iter,
        'time_local_search (s)': time_local_search,
        'total_time (s)': total_time
    })

# Convert the results to a DataFrame for easier visualization
df = pd.DataFrame(results)
pd.set_option('display.max_columns', None)  # Display all columns
# Let's add a column for the product m*n and a column for the growth rate
df['m*n'] = df['input_size_max_iter'] * df['input_size_local_search']
df['growth_rate'] = df['total_time (s)'].pct_change()  # Calculate the growth rate
print(df)


plt.figure(figsize=(10, 6))
plt.plot(df['m*n'], df['total_time (s)'], marker='o', label='Measured time')
plt.xlabel('max_iter * local_search (m*n)')
plt.ylabel('Total runtime (s)')
plt.title('Hill-Climber: Time complexity O(m*n)')
plt.grid(True)
plt.legend()
plt.show()


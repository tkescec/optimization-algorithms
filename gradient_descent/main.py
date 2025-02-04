import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
import sys


# Define the Objective Functions
def quadratic(x):
    """
    Quadratic function in N-dimensional.
    Global minimum at (1, 1, ..., 1) with value 0.

    Args:
        x (numpy.ndarray): N-dimensional input array with shape (N, ...).

    Returns:
        numpy.ndarray: Function value for each input point.
    """
    return np.sum((x - 1) ** 2, axis=0)


def rastrigin(x, A=10):
    """
    Rastrigin function in N-dimensional.
    Global minimum at (0, 0, ..., 0) with value 0.

    Args:
        x (numpy.ndarray): N-dimensional input array with shape (N, ...).
        A (float): Constant value, default is 10.

    Returns:
        numpy.ndarray: Function value for each input point.
    """
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x), axis=0)

# Exact Gradient for Quadratic Function
def quadratic_gradient(x):
    return 2 * (x - 1)

# Exact Gradient for Rastrigin Function (Optional)
def rastrigin_gradient(x, A=10):
    return 2 * x + 2 * A * np.pi * np.sin(2 * np.pi * x)

# Numerical Gradient Estimation
def compute_gradient(f, x, epsilon=1e-5, *args):
    """
    Numerically estimate the gradient of function f at point x using central differences.

    Args:
        f (callable): Objective function.
        x (numpy.ndarray): Current point in 2D space.
        epsilon (float): Small perturbation value.
        *args: Additional arguments for the objective function.

    Returns:
        numpy.ndarray: Gradient vector.
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = np.array(x, copy=True)
        x_minus = np.array(x, copy=True)
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        grad[i] = (f(x_plus, *args) - f(x_minus, *args)) / (2 * epsilon)
    return grad


# 3. Gradient Descent Implementation

def gradient_descent(f, initial_x, learning_rate, num_iterations, *args):
    """
    Perform Gradient Descent optimization.

    Args:
        f (callable): Objective function.
        initial_x (numpy.ndarray): Starting point in 2D space.
        learning_rate (float): Learning rate.
        num_iterations (int): Number of iterations.
        *args: Additional arguments for the objective function.

    Returns:
        tuple: Optimal point, history of points visited, history of function values.
    """
    x = np.array(initial_x, dtype=float)
    history = [x.copy()]
    f_history = [f(x, *args)]
    for i in range(num_iterations):
        grad = compute_gradient(f, x, *args)
        x = x - learning_rate * grad
        history.append(x.copy())
        f_history.append(f(x, *args))
        # Print progress
        print(f"Iteration {i + 1}: x = {x}, f(x) = {f(x, *args)}")
    return x, history, f_history


# Animation Function for 2D Functions
def animate_gradient_descent_2d(f, history, func_name='Rastrigin', A=10,
                                x_range=(-5.12, 5.12), y_range=(-5.12, 5.12),
                                interval=200, save_gif=True, gif_name='gradient_descent.gif'):
    """
    Animate the Gradient Descent process on a 2D function and save as GIF.

    Args:
        f (callable): Objective function.
        history (list): History of points visited by Gradient Descent.
        func_name (str): Name of the function (for title).
        A (float): Constant value for Rastrigin function.
        x_range (tuple): Range for x-axis.
        y_range (tuple): Range for y-axis.
        interval (int): Delay between frames in milliseconds.
        save_gif (bool): Whether to save the animation as GIF.
        gif_name (str): Filename for the saved GIF.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a grid of points
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    y_vals = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Compute Z values
    if func_name.lower() == 'rastrigin':
        Z = f(np.array([X, Y]), A)
    elif func_name.lower() == 'quadratic':
        Z = f(np.array([X, Y]))
    else:
        raise ValueError("Function name not recognized. Use 'Rastrigin' or 'Quadratic'.")

    # Plot contour
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='f(x, y)')

    # Initialize points for Gradient Descent
    gd_points, = ax.plot([], [], 'ro', label='Gradient Descent Steps')
    gd_path, = ax.plot([], [], 'r--', linewidth=1)

    # Set plot limits and labels
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_title(f'Gradient Descent on {func_name} Function')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()

    # Lists to store path
    path_x, path_y = [], []

    # Update function for animation
    def update(frame):
        current_x = history[frame]
        path_x.append(current_x[0])
        path_y.append(current_x[1])
        gd_points.set_data(path_x, path_y)
        gd_path.set_data(path_x, path_y)
        return gd_points, gd_path

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(history),
                                  interval=interval, blit=True, repeat=False)

    # Save as GIF if required
    if save_gif:
        try:
            writer = PillowWriter(fps=1000 // interval)
            ani.save(gif_name, writer=writer)
            print(f"Animation saved as {gif_name}")
        except Exception as e:
            print(f"Error saving animation: {e}")

    plt.show()


# Static Plot for Quadratic Function with Gradient Descent Steps
def plot_gradient_descent_2d(f, history, func_name='Rastrigin', A=10,
                                x_range=(-5.12, 5.12), y_range=(-5.12, 5.12)):
    """
    Plot the Objective function with Gradient Descent steps.

    Args:
        history (list): History of points visited by Gradient Descent.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a grid of points
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    y_vals = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Compute Z values
    if func_name.lower() == 'rastrigin':
        Z = f(np.array([X, Y]), A)
    elif func_name.lower() == 'quadratic':
        Z = f(np.array([X, Y]))
    else:
        raise ValueError("Function name not recognized. Use 'Rastrigin' or 'Quadratic'.")

    # Plot contour
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='f(x, y)')

    # Extract history points
    history = np.array(history)
    path_x = history[:, 0]
    path_y = history[:, 1]

    # Plot Gradient Descent steps
    ax.plot(path_x, path_y, 'ro-', label='Gradient Descent Steps')

    # Mark the minimum
    if func_name.lower() == 'rastrigin':
        ax.plot(0, 0, 'b*', markersize=15, label='Global Minimum')
    elif func_name.lower() == 'quadratic':
        ax.plot(1, 1, 'b*', markersize=15, label='Global Minimum')


    # Set plot limits and labels
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_title(f'Gradient Descent on {func_name} Function')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()
    ax.grid(True)

    plt.show()


# Main Function to Execute the Process
def main():
    """
    Main function to execute Gradient Descent on Quadratic and Rastrigin functions.
    """
    # Optimize Quadratic Function
    print("=== Optimizing Quadratic Function ===")
    quadratic_initial_x = np.array([4.0, 2.0])  # Starting point
    quadratic_learning_rate = 0.1  # Learning rate
    quadratic_num_iterations = 100  # Number of iterations

    # Perform Gradient Descent on Quadratic Function
    quadratic_optimal_x, quadratic_history, f_history = gradient_descent(
        quadratic, quadratic_initial_x, quadratic_learning_rate, quadratic_num_iterations
    )

    print(f"\nQuadratic Function Optimal x found: {quadratic_optimal_x}")
    print(f"Minimum value of Quadratic f(x): {quadratic(quadratic_optimal_x)}")

    # Plot Quadratic Function with Gradient Descent Steps
    plot_gradient_descent_2d(quadratic, quadratic_history, func_name='Quadratic', A=10)
    # Animate Gradient Descent on Quadratic Function
    animate_gradient_descent_2d(
        quadratic, quadratic_history, func_name='Quadratic', A=10,
        gif_name='gradient_descent_quadratic.gif'
    )

    # Optimize Rastrigin Function
    print("\n=== Optimizing Rastrigin Function ===")
    rastrigin_initial_x = np.array([4.0, 4.0])  # Starting point
    rastrigin_learning_rate = 0.1  # Learning rate
    rastrigin_num_iterations = 100  # Number of iterations
    A = 10  # Constant for Rastrigin

    # Perform Gradient Descent on Rastrigin Function
    rastrigin_optimal_x, rastrigin_history, f_history = gradient_descent(
        rastrigin, rastrigin_initial_x, rastrigin_learning_rate, rastrigin_num_iterations, A
    )

    print(f"\nRastrigin Function Optimal x found: {rastrigin_optimal_x}")
    print(f"Minimum value of Rastrigin f(x): {rastrigin(rastrigin_optimal_x, A)}")

    # Plot Rastrigin Function with Gradient Descent Steps
    plot_gradient_descent_2d(rastrigin, rastrigin_history, func_name='Rastrigin', A=A)
    # Animate Gradient Descent on Rastrigin Function
    animate_gradient_descent_2d(
        rastrigin, rastrigin_history, func_name='Rastrigin', A=A,
        gif_name='gradient_descent_rastrigin.gif'
    )


if __name__ == "__main__":
    main()

# Task 16. Experiment basic Gradient Descent
# Run Experiments
def run_experiments(f, initial_x, learning_rates, num_iterations_list, func_name='Quadratic', A=10):
    results = {}
    for lr in learning_rates:
        results[lr] = {}
        for num_iter in num_iterations_list:
            print(f"\n--- Gradient Descent s lr={lr}, iteracije={num_iter} ---")
            optimal_x, history, f_history = gradient_descent(
                f, initial_x, learning_rate=lr, num_iterations=num_iter
            )
            results[lr][num_iter] = {
                'optimal_x': optimal_x,
                'history': history,
                'f_history': f_history
            }
    return results

# Plot Convergence
def plot_convergence(results, learning_rates, num_iterations_list, func_name='Quadratic'):
    plt.figure(figsize=(12, 8))
    for lr in learning_rates:
        for num_iter in num_iterations_list:
            f_history = results[lr][num_iter]['f_history']
            plt.plot(f_history, label=f'lr={lr}, iter={num_iter}')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')
    plt.title(f'Convergence of Gradient Descent on {func_name} Function')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot Optimization Paths
def plot_optimization_paths(f, results, learning_rates, num_iterations_list, initial_x, func_name='Quadratic', A=10):
    plt.figure(figsize=(12, 8))

    # Create a grid of points
    x_vals = np.linspace(-5.12, 5.12, 400)
    y_vals = np.linspace(-5.12, 5.12, 400)
    X, Y = np.meshgrid(x_vals, y_vals)

    if func_name.lower() == 'rastrigin':
        Z = f(np.array([X, Y]), A)
        minimum = np.array([0, 0])
    elif func_name.lower() == 'quadratic':
        Z = f(np.array([X, Y]))
        minimum = np.array([1, 1])
    else:
        raise ValueError("Function name not recognized. Use 'Rastrigin' or 'Quadratic'.")

    # Plot kontura
    contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
    plt.colorbar(contour, label='f(x, y)')

    # Points plot
    for lr in learning_rates:
        for num_iter in num_iterations_list:
            history = np.array(results[lr][num_iter]['history'])
            plt.plot(history[:, 0], history[:, 1], marker='o', markersize=3, label=f'lr={lr}, iter={num_iter}')

    # Mark the minimum
    plt.plot(minimum[0], minimum[1], 'r*', markersize=15, label='Global Minimum')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f' Gradient Descent paths on {func_name} Function')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main Function to Execute Experiments
def main():
    # Experiment Parameters
    learning_rates = [0.01, 0.05, 0.1, 0.2]
    num_iterations_list = [50, 100, 200]

    # Experiment on Quadratic Function
    print("=== Experiment: Quadratic Function ===")
    quadratic_initial_x = np.array([4.0, 2.0])  # Starting point
    quadratic_results = run_experiments(
        f=quadratic,
        initial_x=quadratic_initial_x,
        learning_rates=learning_rates,
        num_iterations_list=num_iterations_list,
        func_name='Quadratic'
    )

    # Convergence plot
    plot_convergence(quadratic_results, learning_rates, num_iterations_list, func_name='Quadratic')

    # Path optimization plot
    plot_optimization_paths(
        f=quadratic,
        results=quadratic_results,
        learning_rates=learning_rates,
        num_iterations_list=num_iterations_list,
        initial_x=quadratic_initial_x,
        func_name='Quadratic'
    )

    # Experiment on Rastrigin Function
    print("\n=== Experiment: Rastrigin Function ===")
    rastrigin_initial_x = np.array([4.0, 4.0]) # Starting point
    A = 10  # Constant for Rastrigin
    rastrigin_results = run_experiments(
        f=rastrigin,
        initial_x=rastrigin_initial_x,
        learning_rates=learning_rates,
        num_iterations_list=num_iterations_list,
        func_name='Rastrigin',
        A=A
    )

    # Convergence plot
    plot_convergence(rastrigin_results, learning_rates, num_iterations_list, func_name='Rastrigin')

    # Path optimization plot
    plot_optimization_paths(
        f=rastrigin,
        results=rastrigin_results,
        learning_rates=learning_rates,
        num_iterations_list=num_iterations_list,
        initial_x=rastrigin_initial_x,
        func_name='Rastrigin',
        A=A
    )


if __name__ == "__main__":
    main()
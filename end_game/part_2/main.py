import numpy as np
import matplotlib.pyplot as plt


def rastrigin(x):
    """
    Rastrigin function:
    f(x) = 10*d + sum(x_i^2 - 10*cos(2*pi*x_i)),
    global minimum: x = 0, f(x) = 0.
    """
    x = np.array(x)
    d = len(x)
    return 10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def firefly_algorithm(fobj, dim, bounds, n=20, beta0=1.0, gamma=1.0, alpha=0.2, max_iter=100):
    """
    Implementation of the Firefly Algorithm for minimization:
      fobj      : objective function that takes a vector x and returns a scalar value
      dim       : problem dimension
      bounds    : (lower_bound, upper_bound) for each vector component
      n         : number of fireflies (population size)
      beta0     : base 'attractiveness'
      gamma     : 'absorption' coefficient (affects attractiveness decay with distance)
      alpha     : random movement factor
      max_iter  : maximum number of iterations
    Returns:
      best      : best found solution vector
      best_fit  : corresponding objective function value
      history   : list of best fitness values per iteration (for convergence visualization)
    """
    lower, upper = bounds

    # 1. Initialize population
    fireflies = np.random.uniform(lower, upper, (n, dim))
    fitness = np.array([fobj(ff) for ff in fireflies])

    # 2. Find initial best solution
    best_idx = np.argmin(fitness)
    best = fireflies[best_idx].copy()
    best_fit = fitness[best_idx]

    history = [best_fit]  # store best fitness value history

    # 3. Main algorithm loop
    for _ in range(max_iter):
        for i in range(n):
            for j in range(n):
                # If firefly j is 'brighter' (better fitness = lower value),
                # then firefly i moves towards j.
                if fitness[j] < fitness[i]:
                    # Distance between fireflies i and j
                    rij = np.linalg.norm(fireflies[i] - fireflies[j])
                    # Compute 'attractiveness' that decreases with distance
                    beta = beta0 * np.exp(-gamma * (rij ** 2))

                    # Move firefly i
                    fireflies[i] = fireflies[i] \
                                   + beta * (fireflies[j] - fireflies[i]) \
                                   + alpha * (np.random.rand(dim) - 0.5)

                    # Restrict solution within the given bounds
                    fireflies[i] = np.clip(fireflies[i], lower, upper)

                    # Update fitness
                    fitness[i] = fobj(fireflies[i])

                    # Check if new solution is the best
                    if fitness[i] < best_fit:
                        best_fit = fitness[i]
                        best = fireflies[i].copy()

        # Store the best fitness value from this iteration (after all updates)
        history.append(best_fit)

    return best, best_fit, history


if __name__ == "__main__":
    # Problem parameters
    dim = 2  # using 2D Rastrigin
    bounds = (-5.12, 5.12)  # range for Rastrigin
    max_iter = 100
    pop_size = 20

    # Call the Firefly Algorithm
    best_sol, best_val, history = firefly_algorithm(
        fobj=rastrigin,
        dim=dim,
        bounds=bounds,
        n=pop_size,
        beta0=1.0,
        gamma=1.0,
        alpha=0.2,
        max_iter=max_iter
    )

    print("Best found solution:", best_sol)
    print("Rastrigin function value at this solution:", best_val)

    # Convergence visualization
    plt.figure(figsize=(8, 5))
    plt.plot(history, label="Firefly - Rastrigin", color="blue")
    plt.title("Convergence of the Firefly Algorithm")
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness (lower is better)")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")  # log-scale for better visualization
    plt.savefig("plots/firefly_convergence.png")
    plt.show()

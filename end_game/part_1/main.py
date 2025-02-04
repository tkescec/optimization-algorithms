import math
import random
import time
import copy
import matplotlib.pyplot as plt

# =====================================
# 1. Data Definition (Problem)
# =====================================

customers = [
    ('Customer1', 5, (1, 2), 2, (0, 10)),
    ('Customer2', 10, (2, 3), 3, (2, 12)),
    ('Customer3', 8, (3, 1), 1, (1, 8)),
    ('Customer4', 6, (4, 4), 2, (3, 15)),
    ('Customer5', 12, (5, 5), 4, (5, 15)),
    ('Customer6', 7, (6, 2), 1, (0, 5)),
    ('Customer7', 15, (7, 8), 3, (5, 20)),
    ('Customer8', 4, (8, 3), 2, (3, 9)),
    ('Customer9', 9, (1, 7), 1, (2, 10)),
    ('Customer10', 11, (4, 6), 2, (4, 14)),
    ('Customer11', 14, (2, 5), 3, (1, 9)),
    ('Customer12', 6, (5, 7), 2, (3, 12)),
    ('Customer13', 10, (3, 3), 4, (2, 10)),
    ('Customer14', 8, (6, 5), 1, (0, 4)),
    ('Customer15', 12, (1, 4), 2, (1, 7)),
    ('Customer16', 5, (4, 2), 3, (3, 8)),
    ('Customer17', 7, (8, 6), 1, (4, 10)),
    ('Customer18', 9, (5, 3), 2, (2, 9)),
    ('Customer19', 11, (3, 6), 2, (4, 11)),
    ('Customer20', 13, (2, 8), 4, (3, 15)),
]

# Depot location
depot = (0, 0)

# Vehicles with different capacities
vehicles = [
    {'vehicle_id': 'Vehicle1', 'capacity': 20},
    {'vehicle_id': 'Vehicle2', 'capacity': 15},
    {'vehicle_id': 'Vehicle3', 'capacity': 25},
]

# Routing rules
routing_rules = {
    'max_distance_per_route': 30,  # Maximum distance per route
    'max_customers_per_route': 5,  # Maximum number of customers per route
}

# Weight of late penalty and capacity penalty
LATE_PENALTY_WEIGHT = 5.0
CAPACITY_PENALTY_WEIGHT = 1000.0


# =====================================
# 2. Helper Functions
# =====================================

def euclidean_distance(p1, p2):
    """Calculates the Euclidean distance between two 2D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def compute_route_cost(route, vehicle, routing_rules, start_time=0):
    """
    Computes the cost of a route for a given vehicle, considering:
      - The traveled distance (including return to depot)
      - Penalties for lateness according to customer time windows
      - Capacity violation of the vehicle

    Parameters:
      - route: list of customer indices (e.g., [0, 3, 7, ...])
      - vehicle: dictionary containing vehicle data (e.g., {'vehicle_id': 'Vehicle1', 'capacity': 20})
      - routing_rules: dictionary containing routing rules
      - start_time: the time at which the vehicle departs from the depot (useful for multiple routes per vehicle)

    Returns:
      - cost: the total cost (distance + penalties)
      - finish_time: the time at which the vehicle returns to the depot after completing the route
    """
    # Check vehicle capacity
    total_demand = sum(customers[cid][1] for cid in route)
    capacity_violation = max(0, total_demand - vehicle['capacity'])
    capacity_penalty = capacity_violation * CAPACITY_PENALTY_WEIGHT

    current_time = start_time
    current_position = depot
    total_distance = 0
    late_penalty = 0

    # Visit customers sequentially
    for cid in route:
        customer_info = customers[cid]
        (x, y) = customer_info[2]
        service_time = customer_info[3]
        (earliest, latest) = customer_info[4]

        # Calculate travel distance to the customer
        travel_dist = euclidean_distance(current_position, (x, y))
        total_distance += travel_dist
        current_time += travel_dist  # travel time

        # If arriving too early, wait until the customer's earliest time
        if current_time < earliest:
            current_time = earliest
        # If arriving too late, apply a penalty
        elif current_time > latest:
            late_penalty += (current_time - latest) * LATE_PENALTY_WEIGHT

        # Add service time and move to the customer's location
        current_time += service_time
        current_position = (x, y)

    # Return to depot
    travel_back = euclidean_distance(current_position, depot)
    total_distance += travel_back
    current_time += travel_back

    # Additional penalty if the route exceeds the maximum allowed distance
    route_distance_penalty = 0
    if total_distance > routing_rules['max_distance_per_route']:
        route_distance_penalty = (total_distance - routing_rules['max_distance_per_route']) * 100

    cost = total_distance + late_penalty + capacity_penalty + route_distance_penalty
    return cost, current_time

def compute_route_details(route, vehicle, routing_rules, start_time=0):
    """
    Computes additional details for a route:
      - finish_time: the total time (including travel, waiting, service, and return to depot)
      - total_distance: the total distance traveled (including return to depot)
      - total_demand: the total demand (capacity used) on the route
    Assumes the vehicle starts from the depot at start_time.
    """
    total_demand = sum(customers[cid][1] for cid in route)
    current_time = start_time
    current_position = depot
    total_distance = 0

    for cid in route:
        (x, y) = customers[cid][2]
        travel_dist = euclidean_distance(current_position, (x, y))
        total_distance += travel_dist
        current_time += travel_dist
        earliest = customers[cid][4][0]
        if current_time < earliest:
            current_time = earliest
        service_time = customers[cid][3]
        current_time += service_time
        current_position = (x, y)
    travel_back = euclidean_distance(current_position, depot)
    total_distance += travel_back
    current_time += travel_back
    return current_time, total_distance, total_demand

def compute_solution_cost(solution, vehicles, routing_rules):
    """
    Computes the total cost of the entire solution.
    The solution is a dictionary where the keys are vehicle_ids and the values are lists of routes.

    For each vehicle:
      - Routes are executed sequentially, with the first route starting at time 0.
      - The finish time (return to depot) of the previous route is used as the start time for the next route.
    """
    total_cost = 0.0

    # Process each vehicle in the solution
    for vehicle_id, routes in solution.items():
        # Retrieve vehicle data
        vehicle = next(v for v in vehicles if v['vehicle_id'] == vehicle_id)
        start_time = 0  # Vehicle departs from depot at time 0
        for route in routes:
            route_cost, finish_time = compute_route_cost(route, vehicle, routing_rules, start_time)
            total_cost += route_cost
            # The next route starts after returning to depot
            start_time = finish_time
    return total_cost

def rebuild_into_routes(flat_solution, routing_rules):
    """
    Reconstructs a flat list of customers into a list of routes,
    enforcing the maximum number of customers per route.
    """
    solution = []
    route = []
    max_customers = routing_rules['max_customers_per_route']
    for cid in flat_solution:
        if len(route) < max_customers:
            route.append(cid)
        else:
            solution.append(route)
            route = [cid]
    if route:
        solution.append(route)
    return solution

def create_random_solution(customers, vehicles, routing_rules):
    """
    Generates an initial solution where all customers are distributed among the vehicles.
    The solution structure is a dictionary with vehicle_id as keys and lists of routes as values.

    In this example:
      - Customers are shuffled randomly.
      - They are distributed among vehicles in a round-robin manner.
      - For each vehicle, the list of customers is split into routes, each containing up to 'max_customers_per_route' customers.
    """
    num_customers = len(customers)
    customer_indices = list(range(num_customers))
    random.shuffle(customer_indices)

    solution = {vehicle['vehicle_id']: [] for vehicle in vehicles}
    # First, distribute customers among vehicles in a round-robin fashion
    vehicle_ids = [vehicle['vehicle_id'] for vehicle in vehicles]
    assigned = {vid: [] for vid in vehicle_ids}

    for i, cid in enumerate(customer_indices):
        vid = vehicle_ids[i % len(vehicle_ids)]
        assigned[vid].append(cid)

    # For each vehicle, split the assigned customers into routes with the max_customers_per_route limit
    for vid in vehicle_ids:
        solution[vid] = rebuild_into_routes(assigned[vid], routing_rules)

    return solution


# =====================================
# 3. Genetic Algorithm (GA)
# =====================================

def initialize_population(pop_size, customers, vehicles, routing_rules):
    """Initializes the population with random solutions."""
    population = []
    for _ in range(pop_size):
        sol = create_random_solution(customers, vehicles, routing_rules)
        population.append(sol)
    return population

def crossover(parent1, parent2, routing_rules):
    """
    Simple crossover:
      - For each vehicle, takes the first part of the routes from parent1 and the rest from parent2 (removing duplicates)
      - Then, it merges the flat list of customers and redistributes them among the vehicles.
    """
    child = {}
    for vehicle_id in parent1.keys():
        # Flatten the routes into a list of customers for the vehicle
        p1_flat = [c for route in parent1[vehicle_id] for c in route]
        p2_flat = [c for route in parent2[vehicle_id] for c in route]

        mid = len(p1_flat) // 2
        child_flat_1 = p1_flat[:mid]
        child_flat_2 = [c for c in p2_flat if c not in child_flat_1]
        child_flat = child_flat_1 + child_flat_2

        # Rebuild the flat list into routes
        child[vehicle_id] = rebuild_into_routes(child_flat, routing_rules)
    return child

def mutation(solution, mutation_rate, routing_rules):
    """
    Mutation: randomly swaps two customers either between different routes or within the same route
    for a randomly selected vehicle.
    """
    if random.random() < mutation_rate:
        vehicle_id = random.choice(list(solution.keys()))
        routes = solution[vehicle_id]
        if len(routes) == 0:
            return solution
        if len(routes) == 1 and len(routes[0]) > 1:
            i, j = random.sample(range(len(routes[0])), 2)
            routes[0][i], routes[0][j] = routes[0][j], routes[0][i]
        elif len(routes) >= 2:
            r1, r2 = random.sample(range(len(routes)), 2)
            if routes[r1] and routes[r2]:
                i = random.randrange(len(routes[r1]))
                j = random.randrange(len(routes[r2]))
                routes[r1][i], routes[r2][j] = routes[r2][j], routes[r1][i]
        solution[vehicle_id] = routes
    return solution

def tournament_selection(fitness_list, size=3):
    """Tournament selection."""
    indices = random.sample(range(len(fitness_list)), size)
    best_i = min(indices, key=lambda i: fitness_list[i])
    return best_i

def genetic_algorithm(customers, vehicles, routing_rules,
                      pop_size=30, max_gens=50, crossover_prob=0.8, mutation_rate=0.1):
    """Main loop of the genetic algorithm."""
    population = initialize_population(pop_size, customers, vehicles, routing_rules)

    best_solution = None
    best_cost = float('inf')

    cost_history = []  # for convergence graph

    for g in range(max_gens):
        fitness_list = []
        for sol in population:
            cost = compute_solution_cost(sol, vehicles, routing_rules)
            fitness_list.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_solution = copy.deepcopy(sol)

        new_population = []

        # Elitism: carry the best solution to the new population
        elite_index = min(range(len(population)), key=lambda i: fitness_list[i])
        elite_solution = copy.deepcopy(population[elite_index])
        new_population.append(elite_solution)

        # Create new offspring
        while len(new_population) < pop_size:
            p1_index = tournament_selection(fitness_list, size=3)
            p2_index = tournament_selection(fitness_list, size=3)

            parent1 = copy.deepcopy(population[p1_index])
            parent2 = copy.deepcopy(population[p2_index])

            if random.random() < crossover_prob:
                child = crossover(parent1, parent2, routing_rules)
            else:
                child = parent1

            child = mutation(child, mutation_rate, routing_rules)
            new_population.append(child)

        population = new_population
        cost_history.append(best_cost)
        print(f"Generation {g}: Best cost = {best_cost}")

    return best_solution, best_cost, cost_history


# =====================================
# 4. Simulated Annealing (SA)
# =====================================

def simulated_annealing(customers, vehicles, routing_rules,
                        initial_temp=100.0, cooling_rate=0.99, max_iter=200):
    """Simulated annealing for VRP using a multi-vehicle solution."""
    current_solution = create_random_solution(customers, vehicles, routing_rules)
    current_cost = compute_solution_cost(current_solution, vehicles, routing_rules)

    best_solution = copy.deepcopy(current_solution)
    best_cost = current_cost

    cost_history = []
    temperature = initial_temp

    for i in range(max_iter):
        neighbor = neighbor_mutation(current_solution, routing_rules)
        neighbor_cost = compute_solution_cost(neighbor, vehicles, routing_rules)

        if accept_move(current_cost, neighbor_cost, temperature):
            current_solution = neighbor
            current_cost = neighbor_cost
            if neighbor_cost < best_cost:
                best_solution = copy.deepcopy(neighbor)
                best_cost = neighbor_cost

        cost_history.append(best_cost)
        temperature *= cooling_rate

    return best_solution, best_cost, cost_history

def neighbor_mutation(solution, routing_rules):
    """
    Generates a neighbor solution for SA for a multi-vehicle VRP.
    It randomly selects one of three types of moves:
      1) Swap two customers within the same route.
      2) Swap two customers between two routes within the same vehicle.
      3) Swap two customers between routes of two different vehicles.
    """
    sol = copy.deepcopy(solution)
    move_type = random.choice([1, 2, 3])

    if move_type == 1:
        # Swap within a route: choose a random vehicle and a random route within it.
        vehicle_id = random.choice(list(sol.keys()))
        if len(sol[vehicle_id]) == 0:
            return sol
        route_index = random.randrange(len(sol[vehicle_id]))
        route = sol[vehicle_id][route_index]
        if len(route) > 1:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
            sol[vehicle_id][route_index] = route

    elif move_type == 2:
        # Swap between two routes within the same vehicle.
        vehicle_id = random.choice(list(sol.keys()))
        routes = sol[vehicle_id]
        if len(routes) < 2:
            if len(routes) == 1 and len(routes[0]) > 1:
                i, j = random.sample(range(len(routes[0])), 2)
                routes[0][i], routes[0][j] = routes[0][j], routes[0][i]
            sol[vehicle_id] = routes
        else:
            r1, r2 = random.sample(range(len(routes)), 2)
            if len(routes[r1]) > 0 and len(routes[r2]) > 0:
                i = random.randrange(len(routes[r1]))
                j = random.randrange(len(routes[r2]))
                routes[r1][i], routes[r2][j] = routes[r2][j], routes[r1][i]
            sol[vehicle_id] = routes

    elif move_type == 3:
        # Swap between routes from two different vehicles.
        vehicle_ids = list(sol.keys())
        if len(vehicle_ids) < 2:
            return sol
        v1, v2 = random.sample(vehicle_ids, 2)
        routes_v1 = sol[v1]
        routes_v2 = sol[v2]
        if routes_v1 and routes_v2:
            r1 = random.randrange(len(routes_v1))
            r2 = random.randrange(len(routes_v2))
            if len(routes_v1[r1]) > 0 and len(routes_v2[r2]) > 0:
                i = random.randrange(len(routes_v1[r1]))
                j = random.randrange(len(routes_v2[r2]))
                routes_v1[r1][i], routes_v2[r2][j] = routes_v2[r2][j], routes_v1[r1][i]
            sol[v1] = routes_v1
            sol[v2] = routes_v2

    for vid in sol.keys():
        flat_list = [c for route in sol[vid] for c in route]
        sol[vid] = rebuild_into_routes(flat_list, routing_rules)

    return sol

def accept_move(current_cost, new_cost, temperature):
    """Metropolis acceptance condition."""
    if new_cost < current_cost:
        return True
    else:
        delta = new_cost - current_cost
        prob = math.exp(-delta / temperature)
        return random.random() < prob


# =====================================
# 5. Hill Climber for Hyperparameters
# =====================================

def hill_climber_for_ga(customers, vehicles, routing_rules,
                        initial_params, step_sizes, max_iterations=10):
    """
    Hill Climber for GA hyperparameters:
      - initial_params = {'pop_size': 30, 'max_gens': 50, 'crossover_prob': 0.8, 'mutation_rate': 0.1}
      - step_sizes = {'pop_size':10, 'max_gens':10, 'crossover_prob':0.1, 'mutation_rate':0.05}
    """
    current_params = copy.deepcopy(initial_params)
    best_solution, best_cost, _ = genetic_algorithm(
        customers, vehicles, routing_rules,
        pop_size=current_params['pop_size'],
        max_gens=current_params['max_gens'],
        crossover_prob=current_params['crossover_prob'],
        mutation_rate=current_params['mutation_rate']
    )

    current_cost = best_cost

    for _ in range(max_iterations):
        param_names = list(current_params.keys())
        random.shuffle(param_names)

        for pname in param_names:
            for direction in [-1, 1]:
                candidate_params = copy.deepcopy(current_params)
                candidate_params[pname] += direction * step_sizes[pname]

                if pname == 'pop_size':
                    candidate_params[pname] = max(candidate_params[pname], 10)
                if pname == 'max_gens':
                    candidate_params[pname] = max(candidate_params[pname], 10)
                if pname in ['crossover_prob', 'mutation_rate']:
                    candidate_params[pname] = max(min(candidate_params[pname], 1.0), 0.0)

                _, cost_candidate, _ = genetic_algorithm(
                    customers, vehicles, routing_rules,
                    pop_size=candidate_params['pop_size'],
                    max_gens=candidate_params['max_gens'],
                    crossover_prob=candidate_params['crossover_prob'],
                    mutation_rate=candidate_params['mutation_rate']
                )

                if cost_candidate < current_cost:
                    current_params = candidate_params
                    current_cost = cost_candidate
                    print(f'Improved to Cost: {current_cost} with Params: {current_params}')

    return current_params, current_cost

def hill_climber_for_sa(customers, vehicles, routing_rules,
                        initial_params, step_sizes, max_iterations=10):
    """
    Hill Climber for SA hyperparameters:
      - initial_params = {'initial_temp':100.0, 'cooling_rate':0.99, 'max_iter':200}
    """
    current_params = copy.deepcopy(initial_params)
    best_solution, best_cost, _ = simulated_annealing(
        customers, vehicles, routing_rules,
        initial_temp=current_params['initial_temp'],
        cooling_rate=current_params['cooling_rate'],
        max_iter=current_params['max_iter']
    )
    current_cost = best_cost

    for _ in range(max_iterations):
        param_names = list(current_params.keys())
        random.shuffle(param_names)

        for pname in param_names:
            for direction in [-1, 1]:
                candidate_params = copy.deepcopy(current_params)

                if pname == 'initial_temp':
                    candidate_params[pname] += direction * step_sizes[pname]
                    candidate_params[pname] = max(1, candidate_params[pname])
                elif pname == 'cooling_rate':
                    candidate_params[pname] += direction * step_sizes[pname]
                    candidate_params[pname] = max(min(candidate_params[pname], 0.999), 0.8)
                elif pname == 'max_iter':
                    candidate_params[pname] += direction * step_sizes[pname]
                    candidate_params[pname] = max(candidate_params[pname], 50)

                _, cost_candidate, _ = simulated_annealing(
                    customers, vehicles, routing_rules,
                    initial_temp=candidate_params['initial_temp'],
                    cooling_rate=candidate_params['cooling_rate'],
                    max_iter=candidate_params['max_iter']
                )

                if cost_candidate < current_cost:
                    current_params = candidate_params
                    current_cost = cost_candidate
                    print(f'Improved to Cost: {current_cost} with Params: {current_params}')

    return current_params, current_cost


# =====================================
# 6. Demonstration and Visualization
# =====================================

def main_demo():
    # 1) Hill Climber for GA
    initial_ga_params = {
        'pop_size': 30,
        'max_gens': 50,
        'crossover_prob': 0.8,
        'mutation_rate': 0.1
    }
    step_sizes_ga = {
        'pop_size': 10,
        'max_gens': 10,
        'crossover_prob': 0.1,
        'mutation_rate': 0.05
    }

    print("** Initial GA parameters:", initial_ga_params)
    best_ga_params, best_ga_cost = hill_climber_for_ga(customers, vehicles, routing_rules,
                                                       initial_ga_params, step_sizes_ga, max_iterations=20)
    print("** Best GA parameters found:", best_ga_params)
    print("** Cost with these parameters:", best_ga_cost)

    # 2) Hill Climber for SA
    initial_sa_params = {
        'initial_temp': 100.0,
        'cooling_rate': 0.99,
        'max_iter': 200
    }
    step_sizes_sa = {
        'initial_temp': 10.0,
        'cooling_rate': 0.01,
        'max_iter': 50
    }
    print("\n** Initial SA parameters:", initial_sa_params)
    best_sa_params, best_sa_cost = hill_climber_for_sa(customers, vehicles, routing_rules,
                                                       initial_sa_params, step_sizes_sa, max_iterations=20)
    print("** Best SA parameters found:", best_sa_params)
    print("** Cost with these parameters:", best_sa_cost)

    # 3) Final comparison GA vs SA with optimized parameters
    print("\n--- Final Comparison ---")
    ga_best_solution, ga_best_cost, ga_history = genetic_algorithm(
        customers, vehicles, routing_rules,
        pop_size=best_ga_params['pop_size'],
        max_gens=best_ga_params['max_gens'],
        crossover_prob=best_ga_params['crossover_prob'],
        mutation_rate=best_ga_params['mutation_rate']
    )

    sa_best_solution, sa_best_cost, sa_history = simulated_annealing(
        customers, vehicles, routing_rules,
        initial_temp=best_sa_params['initial_temp'],
        cooling_rate=best_sa_params['cooling_rate'],
        max_iter=best_sa_params['max_iter']
    )

    print(f"GA Best Cost = {ga_best_cost}")
    print(f"SA Best Cost = {sa_best_cost}")

    # 4) Visualization of convergence with distinct colors
    plt.figure(figsize=(10, 5))
    plt.plot(ga_history, label="Genetic Algorithm", color="navy", linewidth=2)
    plt.plot(sa_history, label="Simulated Annealing", color="darkorange", linewidth=2)
    plt.xlabel("Iteration / Generation")
    plt.ylabel("Best Cost")
    plt.title("Convergence Comparison: GA vs SA")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/convergence.png")
    plt.show()

    # 5) Optional route visualization (e.g., GA and SA solution)
    plot_solution(ga_best_solution, "GA Best Solution")
    plot_solution(sa_best_solution, "SA Best Solution")


def plot_solution(solution, title="Solution"):
    """
    Simple plot of routes on a 2D plane.
    Assumes that the solution is a dictionary where each key is a vehicle_id.
    For each vehicle, routes are plotted sequentially using distinct colors from the 'tab10' colormap.
    Additionally, for each route, the total time to complete the route, the total capacity used,
    and the total distance traveled are annotated on the plot.
    """
    plt.figure(figsize=(12, 12))
    depot_x, depot_y = depot

    # Get a colormap with 10 distinct colors
    cmap = plt.get_cmap("tab10")

    # Plot depot
    plt.scatter(depot_x, depot_y, c="black", marker="s", s=100, label="Depot")

    color_index = 0
    # Plot each vehicle's routes
    for vid, routes in solution.items():
        # Retrieve the corresponding vehicle (for capacity details)
        vehicle = next(v for v in vehicles if v['vehicle_id'] == vid)
        for r_idx, route in enumerate(routes):
            route_coords = [(depot_x, depot_y)]  # start from depot
            for cid in route:
                cx, cy = customers[cid][2]
                route_coords.append((cx, cy))
                plt.text(cx, cy, customers[cid][0], fontsize=9, ha='right')
            route_coords.append((depot_x, depot_y))  # return to depot

            xs = [pt[0] for pt in route_coords]
            ys = [pt[1] for pt in route_coords]

            # Use a color from the colormap; cycle if needed
            color = cmap(color_index % 10)
            color_index += 1

            # Compute route details: finish time, total distance, and total demand.
            finish_time, total_distance, total_demand = compute_route_details(route, vehicle, routing_rules,
                                                                              start_time=0)

            # Plot the route
            plt.plot(xs, ys, color=color, linewidth=2, label=f"{vid} Route {r_idx + 1}")
            plt.scatter(xs, ys, color=color, s=50)

            # Determine a middle point to place the annotation
            mid_idx = len(xs) // 2
            mid_x, mid_y = xs[mid_idx], ys[mid_idx]
            annotation_text = f"Time: {finish_time:.1f}\nDemand: {total_demand}\nDist: {total_distance:.1f}"
            plt.text(mid_x, mid_y, annotation_text, fontsize=10, color="black",
                     bbox=dict(facecolor="white", edgecolor=color, alpha=0.7))

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{title}.png")
    plt.show()


if __name__ == "__main__":
    # Run demonstration
    main_demo()

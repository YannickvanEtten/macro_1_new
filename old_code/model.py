import numpy as np
import help as hp

parameters = {'beta': 0.984, 'sigma': 1, 'alpha': 1/3, 'delta': 0.025, 'epsilon': 0.0001}

beta = parameters['beta']
sigma = parameters['sigma']
alpha = parameters['alpha']
delta = parameters['delta']
epsilon = parameters['epsilon']

# Compute the steady state capital K*
steady_state_capital = hp.compute_steady_state_capital(beta, delta, alpha)
print(steady_state_capital)

init_capital = 0.75 * steady_state_capital

# Make an equally spaced grid between K_1 and K_n
K_grid = hp.make_grid(0.5 * steady_state_capital, 1.5 * steady_state_capital, 1000)

value_function = np.zeros(1000)

payoff_matrix = np.full((1000, 1000), -np.inf)

# Create a payoff matrix (Umat)
for i in range(1000):
    for j in range(1000):
        LB = hp.get_fasible_range_of_K(K_grid[i], alpha, delta)['LB']
        UB = hp.get_fasible_range_of_K(K_grid[i], alpha, delta)['UB']

        if K_grid[j] >= LB and K_grid[j] <= UB:
            payoff_matrix[i, j] = hp.get_utility(hp.get_consumption(K_grid[i], K_grid[j], alpha, delta), sigma)

initial_grid_index = hp.get_index_of_close_value(K_grid, init_capital)

value_function_matrix = np.zeros((1000, 1000))
index_vector = np.zeros(1000)

final_value_function, policy_function = hp.iterate(value_function_matrix, value_function_matrix, payoff_matrix, beta, epsilon, index_vector)
import numpy as np

def compute_steady_state_capital(beta, delta, alpha):
    'returns the steady state capital according to equation (2.12) and given production function'
    return ((1 + beta * (delta - 1)) / (beta * alpha))**(1 / (alpha - 1))

def make_grid(min, max, n):
    'returns a grid of n equally spaced points between min and max'
    return np.linspace(min, max, n)

def get_fasible_range_of_K(current_capital, alpha, delta):
    'returns the LB and UB of the feasible range of capital values for next period'
    return {'LB': (1 - delta) * current_capital, 'UB': current_capital ** alpha + (1 - delta) * current_capital}

def get_consumption(current_capital, next_capital, alpha, delta):
    'returns the consumption given current and next period capital'
    return current_capital ** alpha + (1 - delta) * current_capital - next_capital

def get_utility(consumption, sigma):
    'returns the utility of consumption'
    if (sigma == 1):
        return np.log(consumption)
    return (consumption ** (1 - sigma) - 1) / (1 - sigma)

def get_index_of_close_value(array, value):
    'returns the index of the closest value in the array to the given value'
    return np.abs(array - value).argmin()

def iterate(current_value_function_matrix, prev_value_function_matrix, payoff_matrix, beta, epsilon, index_vector):

    if np.linalg.norm(current_value_function_matrix - prev_value_function_matrix) < epsilon:
        return current_value_function_matrix, index_vector
    
    prev_value_function_matrix = current_value_function_matrix
    
    iteration_matrix = payoff_matrix + beta * current_value_function_matrix

    for i in range(1000):
        row_max = np.amax(iteration_matrix[i])
        index_vector[i] = np.argmax(iteration_matrix[i])

        for j in range(1000):
            current_value_function_matrix[i][j] = row_max

    iterate(current_value_function_matrix, prev_value_function_matrix, payoff_matrix, beta, epsilon, index_vector)
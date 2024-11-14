import numpy as np

matrix = np.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0],
                   [10.0, 11.0, 12.0]])

# Vector to subtract
vector = np.array([0.5, 1.0, 1.5, 2.0])  # Only 4 values

# Reshape vector for broadcasting along columns
#vector = vector[:, np.newaxis]  # Shape becomes (4, 1)

# Subtract the vector from each column
result = matrix - vector
print(result)
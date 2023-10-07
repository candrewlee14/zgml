import numpy as np
import random

# Generate synthetic data points
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 3 * X + 4 + np.random.randn(100, 1)

# Initialize values of m and b
m = random.uniform(-1, 1)
b = random.uniform(-1, 1)

# Hyperparameters
learning_rate = 0.03
n_iterations = 2000

# Perform Stochastic Gradient Descent
for iteration in range(n_iterations):
    random_index = random.randint(0, len(X) - 1)
    xi = X[random_index]
    yi = y[random_index]
    y_pred = m * xi + b
    gradient_m = 2 * xi * (y_pred - yi)
    gradient_b = 2 * (y_pred - yi)
    m -= learning_rate * gradient_m
    b -= learning_rate * gradient_b

# Print the final values of m and b
print("Estimated m:", m)
print("Estimated b:", b)


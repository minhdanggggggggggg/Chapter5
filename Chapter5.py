import numpy as np
import math
import matplotlib.pyplot as plt

# dataset
X1 = np.array([0.245,0.247,0.285,0.299,0.327,0.347,0.356,
0.36,0.363,0.364,0.398,0.4,0.409,0.421,
0.432,0.473,0.509,0.529,0.561,0.569,0.594,
0.638,0.656,0.816,0.853,0.938,1.036,1.045])

X2 = X1**2

y = np.array([0,0,1,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,0,0,1,
1,1,1,1,1,1,1])

# Logistic function
def logistic_function(z):
    return 1 / (1 + np.exp(-z))

# Prediction function
def predict(X1, X2, theta0, theta1, theta2):
    z = theta0 + theta1 * X1 + theta2 * X2
    gz = logistic_function(z)
    return gz

# Cost function
def cost_function(X1, X2, y_true, theta0, theta1, theta2):
    m = len(X1)
    epsilon = 1e-15
    y_pred = predict(X1, X2, theta0, theta1, theta2)
    cost = - (1/m) * np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    reg_term = (lambda_reg / (2*m)) * (theta1**2 + theta2**2)
    cost += reg_term
    return cost

# Gradient Descent
def gradient_descent(X1, X2, y, theta0, theta1, theta2, learning_rate, lambda_reg):
    m = len(X1)

    gradient0 = (1/m) * np.sum(predict(X1, X2, theta0, theta1, theta2) - y)
    gradient1 = (1/m) * (np.sum(predict(X1, X2, theta0, theta1, theta2) - y) * X1 + lambda_reg * theta1)
    gradient2 = (1/m) * (np.sum(predict(X1, X2, theta0, theta1, theta2) - y) * X2 + lambda_reg * theta2)

    new_theta0 = theta0 - learning_rate * gradient0
    new_theta1 = theta1 * (1 - learning_rate * lambda_reg / m) - learning_rate * gradient1
    new_theta2 = theta2 * (1 - learning_rate * lambda_reg / m) - learning_rate * gradient2

    return new_theta0, new_theta1, new_theta2

# Theta0, Theta1, learning_rate
np.random.seed()
theta0 = np.random.rand()
theta1 = np.random.rand()
theta2 = np.random.rand()
learning_rate = 1e-5
lambda_reg = 1
iterations = 100

for i in range(iterations):
    theta0, theta1, theta2 = gradient_descent(X1, X2, y, theta0, theta1, theta2, learning_rate, lambda_reg)
    cost = cost_function(X1, X2, y, theta0, theta1, theta2)

print(f"theta0 = {theta0}")
print(f"theta1 = {theta1}")
print(f"theta2 = {theta2}")
print(f"Cost = {cost}")

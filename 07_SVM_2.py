import numpy as np
from scipy.optimize import minimize

X = np.array([[3, 3], [4, 3], [1, 1]])
y = np.array([1, 1, -1])
k = np.dot(X, X.T)


def objective(alpha):
    return -np.sum(alpha) + 0.5 * np.sum(np.outer(y, y) * np.outer(alpha, alpha) * k)


def constraint1(alpha):
    return alpha


def constraint2(alpha):
    return np.dot(alpha, y)


N = X.shape[0]
bounds = [(0, None)] * N
constraints = [{'type': 'ineq', 'fun': constraint1}, {'type': 'eq', 'fun': constraint2}]

alpha_initial = np.zeros(N)
result = minimize(objective, alpha_initial, bounds=bounds, constraints=constraints)

alphas = result.x
print(f'Optimal alphas:[{alphas[0]},{alphas[1]},{alphas[2]}]')

w = np.dot(alphas * y, X)
print(w)

b_index = np.argmax(alphas)
b = y[b_index] - np.sum(alphas * y * np.dot(X, X[b_index]))
print(b)

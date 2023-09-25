import numpy as np
import matplotlib.pyplot as plt

w1 = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
w2 = np.array([[4, 4], [6, 4], [6, 6], [4, 6]])

m1 = np.mean(w1, axis=0)
m2 = np.mean(w2, axis=0)

cov1 = np.cov(w1.T, bias=True)
cov2 = np.cov(w2.T, bias=True)

inv_cov1 = np.linalg.inv(cov1)
inv_cov2 = np.linalg.inv(cov2)

w = np.dot((m1 - m2), inv_cov1)
b = -0.5 * np.dot(m1, np.dot(inv_cov1, m1)) + 0.5 * np.dot(m2, np.dot(inv_cov2, m2))


def discriminant_function(_x):
    return -(w[0] * _x + b) / w[1]


x = np.linspace(-2, 7, 100)
plt.plot(x, discriminant_function(x), 'r-', label='Discriminant')
plt.scatter(w1[:, 0], w1[:, 1], c='blue', label='ω1')
plt.scatter(w2[:, 0], w2[:, 1], c='green', label='ω2')
plt.legend()

plt.show()

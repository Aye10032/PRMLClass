from fractions import Fraction

import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(formatter={'all': lambda x: str(Fraction(x).limit_denominator())})


x = np.array([
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0]],
    [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 1]]
])
p = np.array([1 / 2, 1 / 2])

m = np.mean(x, axis=1)
m0 = np.zeros(3)
for i in range(2):
    m0 = np.add(m0, m[i] * p[i])

z = x - m0

R = np.zeros((3, 3))
for i in range(z.shape[0]):
    R += p[i] * np.mean(np.einsum('ij,ik->ijk', z[i], z[i]), axis=0)

eig_val, eig_vec = np.linalg.eig(R)

sort_indices = np.argsort(-eig_val)
eig_val = eig_val[sort_indices]
eig_vec = eig_vec[sort_indices, :]


# 降到2维
transform = eig_vec[:2].T

y = np.dot(z, transform)

class1_data = y[0]
class2_data = y[1]


plt.scatter(class1_data[:, 0], class1_data[:, 1], marker='^', color='red', alpha=0.5, label='class 1')
plt.scatter(class2_data[:, 0], class2_data[:, 1], marker='o', color='blue', alpha=0.5, label='class 2')

plt.title('KL Transformation 2D')
plt.legend()
plt.show()


# 降到1维
transform = eig_vec[:1].T

y = np.dot(z, transform)

class1_data = y[0]
class2_data = y[1]


plt.scatter(class1_data, np.zeros_like(class1_data), marker='^', color='red', alpha=0.5, label='class 1')
plt.scatter(class2_data, np.zeros_like(class2_data), marker='o', color='blue', alpha=0.5, label='class 2')

plt.title('KL Transformation 1D')
plt.legend()
plt.show()

from fractions import Fraction

import numpy as np

np.set_printoptions(formatter={'all': lambda x: str(Fraction(x).limit_denominator())})

w = np.array([
    [[1, 0], [2, 0], [1, 1]],
    [[-1, 0], [0, 1], [-1, 1]],
    [[-1, -1], [0, -1], [0, -2]]
])

m = np.mean(w, axis=1)
m0 = np.mean(m, axis=0)

C = np.zeros((3, 2, 2))
for i in range(3):
    diff = w[i] - m[i]
    C[i] = np.dot(diff.T, diff)
C = (1 / 3) * C

sw = np.zeros((2, 2))
for i in range(3):
    sw = np.add(sw, (1 / 3) * C[i])
print('Sw=')
print(sw)

sb = np.zeros((2, 2))
for i in range(3):
    sb = np.add(sb, np.outer((m[i] - m0), (m[i] - m0)))
sb = (1 / 3) * sb
print('Sb = ')
print(sb)

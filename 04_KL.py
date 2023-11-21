import numpy as np

w1 = np.array([[1, 0], [2, 0], [1, 1]])
w2 = np.array([[-1, 0], [0, 1], [-1, 1]])
w3 = np.array([[-1, -1], [0, -1], [0, -2]])

cov = np.zeros((3, 2, 2))
cov[0] = np.cov(w1.T, bias=True)
cov[1] = np.cov(w2.T, bias=True)
cov[2] = np.cov(w3.T, bias=True)

sw = np.zeros((2, 2))
for i in range(3):
    sw = np.add(sw, cov[i])
sw = (1 / 3) * sw
print('S_w = ')
print(sw)

m = np.zeros((3, 2))
m[0] = np.mean(w1, axis=0)
m[1] = np.mean(w2, axis=0)
m[2] = np.mean(w3, axis=0)
m0 = np.mean(m, axis=0)

sb = np.zeros((2, 2))
for i in range(3):
    sb = np.add(sb, np.outer((m[i] - m0), (m[i] - m0)))
sb = (1 / 3) * sb
print('S_b = ')
print(sb)

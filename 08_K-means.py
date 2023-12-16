import numpy as np

X = np.array([
    [5.9, 3.2],
    [4.6, 2.9],
    [6.2, 2.8],
    [4.7, 3.2],
    [5.5, 4.2],
    [5.0, 3.0],
    [4.9, 3.1],
    [6.7, 3.1],
    [5.1, 3.8],
    [6.0, 3.0]
])

mu = np.array([
    [6.2, 3.2],
    [6.6, 3.7],
    [6.5, 3.0],
])

distances = np.linalg.norm(X[:, np.newaxis] - mu, axis=-1)

labels = np.argmin(distances, axis=-1)

new_mu = np.array([np.mean(X[labels == i], axis=0) for i in range(mu.shape[0])])

cluster1_samples = X[labels == 0]
new_mu_1 = new_mu[0]

print('属于第一簇的样本：')
print(cluster1_samples)
print(f'更新后的第一簇中心：{np.round(new_mu_1, 2)}')

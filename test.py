import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from mpl_toolkits.mplot3d import Axes3D

# 定义两个模式
w1 = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 0], [1, 1, 0]])
w2 = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1]])

# 创建标签
y = np.concatenate((np.zeros(len(w1)), np.ones(len(w2))))

# 创建数据集
X = np.concatenate((w1, w2))

# 创建贝叶斯分类器对象
gnb = GaussianNB()

# 拟合数据
gnb.fit(X, y)

# 绘制判别界面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制数据点
ax.scatter(w1[:, 0], w1[:, 1], w1[:, 2], c='red', label='w1')
ax.scatter(w2[:, 0], w2[:, 1], w2[:, 2], c='blue', label='w2')

# 计算判别界面上的点
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
xx, yy = np.meshgrid(x, y)
zz = np.linspace(0, 1, 10)
xx, yy, zz = np.meshgrid(x, y, zz)
z = gnb.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
z = z.reshape(xx.shape)

# 绘制判别界面
ax.scatter(xx[z == 0], yy[z == 0], zz[z == 0], c='red', alpha=0.2)
ax.scatter(xx[z == 1], yy[z == 1], zz[z == 1], c='blue', alpha=0.2)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('Discriminant Boundary')

plt.legend()
plt.show()
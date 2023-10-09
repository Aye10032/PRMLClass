import numpy as np
import matplotlib.pyplot as plt


def d12(_x):
    return -2 * _x + 1


def d13(_x):
    return 2 * _x - 1


def d23(_x):
    return 0 * np.ones_like(_x)


x = np.arange(-10, 10, 0.1)

# 绘制图像
plt.plot(x, d12(x), label='$d_{12}(x) = -2x_1-x_2+1=0$', color='green')
plt.plot(x, d13(x), label='$d_{13}(x)=-2x_1+x_2+1$', color='blue')
plt.plot(x, d23(x), label='$d_{23}(x)=0$', color='orange')

plt.fill_between(x, d12(x), d13(x), where=(x <= 0.5), color='green', alpha=0.2)
plt.fill_between(x, np.maximum(d12(x), d23(x)), 10 * np.ones_like(x), color='blue', alpha=0.2)
plt.fill_between(x, np.minimum(d13(x), d23(x)), -10 * np.ones_like(x), color='orange', alpha=0.2)
#
plt.annotate('$\\omega_1$', xy=(-5, 1), xytext=(-5, 1), fontsize=12, color='black')
plt.annotate('$\\omega_2$', xy=(6, 2), xytext=(6, 2), fontsize=12, color='black')
plt.annotate('$\\omega_3$', xy=(6, -2), xytext=(6, -2), fontsize=12, color='black')

# 添加标题和图例
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.title('多类情况3')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.grid(True)
plt.legend(loc='lower right')

# 显示图像
plt.show()

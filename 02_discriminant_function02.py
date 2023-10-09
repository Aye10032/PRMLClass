import numpy as np
import matplotlib.pyplot as plt


def d13(_x):
    return -_x + 1


def d23(_x):
    return _x - 1


x = np.arange(-10, 10, 0.1)

# 绘制图像
plt.axvline(x=0, label='$d_{12}(x)=-x_1$', color='green')
plt.plot(x, d13(x), label='$d_{13}(x)=x_1+x_2-1$', color='blue')
plt.plot(x, d23(x), label='$d_{23}(x)=x_1-x_2-1$', color='orange')

plt.fill_between(x, 10 * np.ones_like(x), d13(x), where=(x <= 0), color='green', alpha=0.2)
plt.fill_between(x, d23(x), -9 * np.ones_like(x), where=(x >= 0), color='blue', alpha=0.2)
plt.fill_between(x, d23(x), d13(x), where=(x <= 1), color='orange', alpha=0.2)

plt.annotate('$\\omega_1$', xy=(-4, 6), xytext=(-4, 6), fontsize=12, color='black')
plt.annotate('$\\omega_2$', xy=(6, 0), xytext=(6, 0), fontsize=12, color='black')
plt.annotate('$\\omega_3$', xy=(-5, 0), xytext=(-5, 0), fontsize=12, color='black')

# 添加标题和图例
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.title('多类情况2')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(-10, 10)
plt.ylim(-9, 10)
plt.grid(True)
plt.legend(loc='lower right')

# 显示图像
plt.show()

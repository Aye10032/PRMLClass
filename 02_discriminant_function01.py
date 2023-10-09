import numpy as np
import matplotlib.pyplot as plt


def d2(_x):
    return -_x + 1


def d3(_x):
    return _x - 1


x = np.arange(-10, 10, 0.1)

# 绘制图像
plt.axvline(x=0, label='$d_1(x)=-x_1$', color='green')
plt.plot(x, d2(x), label='$d_2(x)=x_1+x_2-1$', color='blue')
plt.plot(x, d3(x), label='$d_3(x)=x_1-x_2-1$', color='orange')

plt.fill_between(x, d2(x), d3(x), where=(x <= 0), color='green', alpha=0.2)
plt.fill_between(x, 10 * np.ones_like(x), np.maximum(d2(x), d3(x)), where=(x >= 0), color='blue', alpha=0.2)
plt.fill_between(x, np.minimum(d2(x), d3(x)), -9 * np.ones_like(x), where=(x >= 0), color='orange', alpha=0.2)

plt.annotate('$\\omega_1$', xy=(-6, 0), xytext=(-6, 0), fontsize=12, color='black')
plt.annotate('$\\omega_2$', xy=(4, 6), xytext=(4, 6), fontsize=12, color='black')
plt.annotate('$\\omega_3$', xy=(4, -6), xytext=(4, -6), fontsize=12, color='black')

# 添加标题和图例
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.title('多类情况1')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(-10, 10)
plt.ylim(-9, 10)
plt.grid(True)
plt.legend(loc='right')

# 显示图像
plt.show()

import numpy as np
import matplotlib.pyplot as plt


# 物理常数
kB = 1.380649e-23  # 玻尔兹曼常数，单位：J/K

# 样本参数
V = 1000e-6  # 体积，1000立方厘米转换为立方米
rho = 6.022e28  # 原子数密度，单位：m^-3
theta_D = 428  # 德拜温度，单位：K


def integrand(x):
    """被积函数：x^4 * e^x / (e^x - 1)^2

    参数：
    x : float 或 numpy.ndarray
        积分变量

    返回：
    float 或 numpy.ndarray：被积函数的值
    """
    # 在这里实现被积函数
    if isinstance(x, np.ndarray):
        return np.array([integrand(xi) for xi in x])

        # 对于单个值的计算
    exp_x = np.exp(x)
    return x**4 * exp_x / (exp_x - 1)**2


def gauss_quadrature(f, a, b, n):
    """实现高斯-勒让德积分

    参数：
    f : callable
        被积函数
    a, b : float
        积分区间的端点
    n : int
        高斯点的数量

    返回：
    float：积分结果
    """
    # 在这里实现高斯积分
    # 获取高斯-勒让德求积的节点和权重
    x, w = np.polynomial.legendre.leggauss(n)

    # 将[-1,1]区间映射到[a,b]区间
    t = 0.5 * (x + 1) * (b - a) + a

    # 计算积分
    return 0.5 * (b - a) * np.sum(w * f(t))


def cv(T):
    """计算给定温度T下的热容

    参数：
    T : float
        温度，单位：K

    返回：
    float：热容值，单位：J/K
    """
    # 在这里实现热容计算
    u= theta_D / T

    # 使用高斯积分计算
    i=gauss_quadrature(integrand, 0, u, 50)

    # 计算热容
    r=9 * V * rho * kB * (T / theta_D) ** 3 * i
    return r,i,u


def plot_cv():
    """绘制热容随温度的变化曲线"""
    # 在这里实现绘图功能
    # 生成温度点（使用线性间距）
    T = np.linspace(5, 500, 200)
    C_V = []
    # 计算对应的热容值
    for t in T:
        r, _, _ = cv(t)
        C_V.append(r)
    C_V = np.array(C_V)

    # 创建图表
    plt.figure(figsize=(10, 6))

    # 绘制热容曲线
    plt.plot(T, C_V, 'b-', label='Debye Model')

    # 添加参考线
    # 低温T^3行为
    T_low = np.linspace(5, 50, 50)
    r,i,u=cv(50)
    C_low = r * (T_low / 50) ** 3
    plt.plot(T_low, C_low, 'r--', label='T³ Law')

    # Add labels and title
    plt.xlabel('Temperature (K)')
    plt.ylabel('Heat Capacity (J/K)')
    plt.title('Solid Heat Capacity vs Temperature (Debye Model)')

    # Add grid
    plt.grid(True, which='both', ls='-', alpha=0.2)

    # Add legend
    plt.legend()

    # 显示图表
    plt.show()


def test_cv():
    """测试热容计算函数"""
    # 测试一些特征温度点的热容值
    test_temperatures = [5, 50,100, 300, 500,1000]
    print("\n测试不同温度下的热容值：")
    print("-" * 40)
    print("温度 (K)\t热容 (J/K)\t积分值\t积分上限")
    print("-" * 40)
    for T in test_temperatures:
        f,s,t=cv(T)
        print(f"{T:8.1f}\t{f:10.3e}\t{s:10.3e}\t{t:10.3e}")


def main():
    # 运行测试
    test_cv()

    # 绘制热容曲线
    plot_cv()


if __name__ == '__main__':
    main()

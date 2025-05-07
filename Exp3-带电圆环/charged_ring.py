import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad  # 导入数值积分函数

# --- 常量定义 ---
a = 1.0  # 圆环半径 (单位：米)
q = 1.0  # 电荷参数 (总电荷 Q = 4*pi*eps0*q)
eps0 = 1.0  # 真空介电常数 (为简化计算设为1)
C = q / (2 * np.pi)  # 电势计算公式中的常数项

# --- 计算函数 ---

def potential_integrand(phi, y, z):
    """
    电势积分的被积函数
    参数:
        phi: 圆环上的角度参数 (弧度)
        y, z: 计算点的y,z坐标
    返回:
        被积函数值 (1/R)
    """
    # 计算场点到圆环上某点的距离
    R = np.sqrt((a * np.cos(phi))**2 + (y - a * np.sin(phi))**2 + z**2)
    return 1.0 / R if R > 1e-10 else 0  # 避免除零错误

def calculate_potential(y, z):
    """
    计算单点(0,y,z)处的电势
    参数:
        y, z: 计算点的y,z坐标
    返回:
        该点的电势值
    """
    # 使用quad函数进行0到2π的积分
    result, _ = quad(potential_integrand, 0, 2*np.pi, args=(y, z))
    return C * result  # 乘以常数项得到最终电势

def calculate_potential_on_grid(y_coords, z_coords):
    """
    在yz平面网格上计算电势分布
    参数:
        y_coords: y坐标数组
        z_coords: z坐标数组
    返回:
        V: 电势矩阵
        y_grid, z_grid: 网格坐标
    """
    # 初始化电势矩阵
    V = np.zeros((len(z_coords), len(y_coords)))
    
    # 遍历网格中的每个点
    for i, z in enumerate(z_coords):
        for j, y in enumerate(y_coords):
            V[i,j] = calculate_potential(y, z)
    
    # 生成网格坐标
    return V, *np.meshgrid(y_coords, z_coords)

def calculate_electric_field_on_grid(V, y_coords, z_coords):
    """
    通过电势梯度计算电场
    参数:
        V: 电势矩阵
        y_coords: y坐标数组
        z_coords: z坐标数组
    返回:
        Ey, Ez: 电场的y和z分量
    """
    # 计算网格间距
    dy = y_coords[1] - y_coords[0]
    dz = z_coords[1] - z_coords[0]
    
    # 计算电势梯度(注意负号)
    Ey, Ez = np.gradient(-V, dy, dz)
    
    return Ey, Ez

# --- 可视化函数 ---

def plot_potential_and_field(y_coords, z_coords, V, Ey, Ez, y_grid, z_grid):
    """
    绘制电势和电场分布图
    参数:
        y_coords, z_coords: 坐标范围
        V: 电势矩阵
        Ey, Ez: 电场分量
        y_grid, z_grid: 网格坐标
    """
    fig = plt.figure(figsize=(14, 6))  # 创建画布
    
    # 1. 绘制等势线图
    ax1 = plt.subplot(1, 2, 1)
    
    # 设置等势线层级
    levels = np.linspace(V.min(), V.max(), 20)
    
    # 绘制填充等势线
    contour = ax1.contourf(y_grid, z_grid, V, levels=levels, cmap='plasma')
    plt.colorbar(contour, label='电势 (V)')
    
    # 设置坐标轴标签和标题
    ax1.set_xlabel('y [a]')
    ax1.set_ylabel('z [a]')
    ax1.set_title('Equipotential Lines in yz-plane')
    ax1.set_aspect('equal')  # 保持纵横比
    ax1.grid(True, alpha=0.3)  # 添加半透明网格
    
    # 标记圆环位置
    ax1.plot([-a, a], [0, 0], 'ro', markersize=5, label='ring')
    ax1.legend()
    
    # 2. 绘制电场线图
    ax2 = plt.subplot(1, 2, 2)
    
    # 计算电场强度大小(用于着色)
    E_magnitude = np.sqrt(Ey**2 + Ez**2)
    
    # 绘制流线图(电场线)
    ax2.streamplot(y_grid, z_grid, Ey, Ez, 
                  color=E_magnitude,  # 按场强着色
                  cmap='viridis',     # 使用viridis颜色映射
                  linewidth=1,        # 线宽
                  density=2,          # 流线密度
                  arrowstyle='->')    # 箭头样式
    
    # 添加颜色条
    plt.colorbar(ax2.collections[0], label='电场强度大小')
    
    # 设置坐标轴标签和标题
    ax2.set_xlabel('y [a]')
    ax2.set_ylabel('z [a]')
    ax2.set_title('Electric Field Lines in yz-plane')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.plot([-a, a], [0, 0], 'ro', markersize=5)
    
    plt.tight_layout()  # 自动调整子图间距
    plt.show()

# --- 主程序 ---
if __name__ == "__main__":
    # 定义计算区域 (yz平面, x=0)
    y_range = np.linspace(-2*a, 2*a, 40)  # y方向坐标
    z_range = np.linspace(-2*a, 2*a, 40)  # z方向坐标
    
    print("正在计算电势分布...")
    V, y_grid, z_grid = calculate_potential_on_grid(y_range, z_range)
    
    print("正在计算电场分布...")
    Ey, Ez = calculate_electric_field_on_grid(V, y_range, z_range)
    
    print("正在绘制结果...")
    plot_potential_and_field(y_range, z_range, V, Ey, Ez, y_grid, z_grid)
    print("计算和绘图完成。")

# -*- coding: utf-8 -*-
"""
 Created on Thu May 18 2023 22:03:50
 Modified on 2023-5-18 22:03:50
 
 @auther: HJ https://github.com/zhaohaojie1998
"""
#


''' 无人机轨迹跟踪控制 demo '''
import numpy as np
from copy import deepcopy
from math import cos, sin, atan, pi
from scipy.integrate import odeint
g = 9.8


#----------------------------- ↓↓↓↓↓ 飞行参数设置 ↓↓↓↓↓ ------------------------------#
# 状态约束
H_LIM = np.array([0, 15000])       # 高度区间
V_LIM = np.array([80, 500])        # 速度区间

# 控制约束
Ts = 0.05                           # 采样时间 (控制周期)
NX_LIM = np.array([-1, 2])         # 切向过载 nx 约束
NY_LIM = np.array([0, 5])          # 法向过载 ny 约束
W_LIM = np.array([-pi/2, pi/2])*Ts # 滚转角速度 dμ 约束, 1s最大滚90°

# 初始状态
STATE_INIT = [0., 10., 0., 200., 0., 0.]



#----------------------------- ↓↓↓↓↓ 飞行动力学模型 ↓↓↓↓↓ ------------------------------#
ODE_TIMES = 2  # 一个dt区间积分几次
class FixedWingModel:
    """固定翼过载控制模型(面对称飞行器)\n
    地面坐标系, Oy指天\n
    s = [x, y, z, V, θ, ψv]\n
    u = [nx2, ny3, μ], μ为速度滚转角(gamma_v)\n
    """

    def __init__(self, s_init, dt=Ts):
        self.dt = dt
        self.t = 0
        self.u = np.zeros(3, dtype=np.float32)
        self.s = np.array(s_init) #! 初始化状态
    
    def __call__(self, u):
        # 数值积分
        t = np.linspace(0.0, self.dt, ODE_TIMES)
        s_new = odeint(self.ode_model, self.s, t, args=(u, )) # (len(t), len(s))
        x, y, z, V, θ, ψ = s_new[-1]

        # 物理边界
        y = np.clip(y, H_LIM[0], H_LIM[1]) # [y_low, y_high]
        V = np.clip(V, V_LIM[0], V_LIM[1]) # [V_low, V_high]
        θ = self.limit_angle(θ)            # (-π, π] 
        ψ = self.limit_angle(ψ)            # (-π, π]

        # 更新状态
        self.t += self.dt
        self.u = u
        self.s = np.array([x, y, z, V, θ, ψ])
        return self.position

    @property
    def states(self):
        """无人机状态"""
        return self.s
    
    @property
    def position(self):
        """无人机位置"""
        return self.s[:3]
    
    @property
    def control(self):
        """无人机过载控制"""
        return self.u

    @staticmethod
    def ode_model(s, t, u):
        """
        >>> dx/dt = V cos(θ) cos(ψ)
        >>> dy/dt = V sin(θ)
        >>> dz/dt = -V cos(θ) sin(ψ)
        >>> dV/dt = g * (nx - sin(θ))
        >>> dθ/dt = g / V * (ny * cos(μ) - cos(θ))
        >>> dψ/dt = -g * ny * sin(μ) / V / cos(θ)
        """
        V, θ, ψ = s[3:] # s <- (6,)
        nx, ny, μ = u

        if np.abs(cos(θ)) < 0.01: #! θ = 90° 没法积分了!!!
            dψ = 0 
        else:
            dψ = -g * ny * sin(μ) / (V * cos(θ))

        dsdt = [
            V * cos(θ) * cos(ψ),
            V * sin(θ),
            -V * cos(θ) * sin(ψ),
            g * (nx - sin(θ)),
            g / V * (ny*cos(μ) - cos(θ)),
            dψ,
        ]
        return dsdt
    
    @staticmethod
    def limit_angle(x, mode=1):
        """ 
        mode1 : (-inf, inf) -> (-π, π] 
        mode2 : (-inf, inf) -> [0, 2π)
        """
        x = x - x//(2*pi) * 2*pi # any -> [0, 2π)
        if mode == 1 and x > pi:
            return x - 2*pi      # [0, 2π) -> (-π, π]
        return x
    


#----------------------------- ↓↓↓↓↓ 飞行器位置偏差控制模型 ↓↓↓↓↓ ------------------------------#
def ControlModel(a, u_last):
    """加速度转换成过载 (PID/ADRC位置误差输出的控制量可以理解为需要提供的加速度)"""
    ax, ay, az = a
    nx, ny, μ = deepcopy(u_last)

    dμ = np.clip(-atan(az / (g*ny + ay + 1e-8)) , W_LIM[0], W_LIM[1]) # [w_min, w_max]
    dnx = ax / g
    dny = (g*ny + ay) / (g*cos(dμ) + 1e-8) - ny

    μ = FixedWingModel.limit_angle(μ+dμ)       # (-π, π]
    nx = np.clip(nx+dnx, NX_LIM[0], NX_LIM[1]) # [nx_min, nx_max]
    ny = np.clip(ny+dny, NY_LIM[0], NY_LIM[1]) # [ny_min, ny_max]

    return np.array([nx, ny, μ])
















# DEBUG:
if __name__ == '__main__':
    from mpl_toolkits import mplot3d
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # 绘图数据存储
    class Data:
        pass
    data = Data()
    data.ti = []
    data.xi = []
    data.yi = []
    data.zi = []
    data.Vi = []
    data.θi = []
    data.ψi = []
    data.ui = []
    data.tr = []
    data.xr = []
    data.yr = []
    data.zr = []
    data.Vr = []
    data.θr = []
    data.ψr = []
    data.ur = []

    # 初始化制导模型
    t = 0.0
    tf = 150.0
    plant = FixedWingModel(STATE_INIT)

    # 理想制导律设计
    u0 = [0, 2.001, pi/3] # 螺旋升天
    u1 = [0, 1, 0]      # 匀速直线飞行
    u2 = [0.5, 1, 0]    # 加速直线飞行
    u3 = [-0.5, 1, 0]   # 减速直线飞行
    
    # 理想轨迹生成
    v_list = []
    while t <= tf:
        if t < 5:
            u = u2
        elif t < 10:
            u = u3
        elif t < 20:
            u = u1
        else: 
            u = u0
        plant(u)
        t += Ts
        v_list.append(plant.position)
        
        # log
        data.ti.append(plant.t)
        data.xi.append(plant.s[0])
        data.yi.append(plant.s[1])
        data.zi.append(plant.s[2])
        data.Vi.append(plant.s[3])
        data.θi.append(plant.s[4])
        data.ψi.append(plant.s[5])
        data.ui.append(plant.u)


    # 无人机轨迹跟踪控制
    plant = FixedWingModel(STATE_INIT)
    from ctrl import PID, PIDConfig, ADRC, ADRCConfig
    cfg = PIDConfig(
        Ts, dim=3, 
        Kp=[10,500,500], 
        Ki=[10,1000,1000], 
        Kd=[100000,100000,100000], 
        u_max=[30, 30, 30],
        u_min=[-30, -30, -30],
        Kaw=0.3,
        max_err=1000)
    ctr = PID(cfg)
    # cfg = ADRCConfig(
    #     Ts, 3,
    #     r = [100., 1000., 1000.],           # 快速跟踪因子 (float or list)
    #     # 扩张状态观测器
    #     b0 = 1000.,          # 被控系统系数 (float or list)
    #     delta = 0.015,      # fal(e, alpha, delta)函数线性区间宽度 (float or list)
    #     beta01 = 1500.,      # ESO反馈增益1 (float or list)
    #     beta02 = 2500.,      # ESO反馈增益2 (float or list)
    #     beta03 = 5500.,      # ESO反馈增益3 (float or list)
    #     # 非线性状态反馈控制率
    #     alpha1 = 200/201,   # 0 < alpha1 < 1  (float or list)
    #     alpha2 = 201/200,   # alpha2 > 1      (float or list)
    #     beta1 = [10., 1., 1.],        # 跟踪输入信号增益 (float or list)
    #     beta2 = [0.0009, 90, 9],     # 跟踪微分信号增益 (float or list)
    #     )
    # ctr = ADRC(cfg)
    
    for v in v_list:
        a = ctr(v, plant.position)
        u = ControlModel(a, plant.control)
        plant(u)

        # log
        data.tr.append(plant.t)
        data.xr.append(plant.s[0])
        data.yr.append(plant.s[1])
        data.zr.append(plant.s[2])
        data.Vr.append(plant.s[3])
        data.θr.append(plant.s[4])
        data.ψr.append(plant.s[5])
        data.ur.append(plant.u)

    ctr.show()






    # 绘图
    mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 修复字体bug
    mpl.rcParams['axes.unicode_minus'] = False            # 用来正常显示负号
    mpl.rcParams['figure.figsize'] = [15, 12]             # 视图窗口大小，英寸表示(默认[6.4,4.8])
    mpl.rcParams['font.size'] = 15                        # 所有字体大小(默认12)
    mpl.rcParams['xtick.labelsize'] = 12                  # X轴刻度字体大小(3D->XYZ)
    mpl.rcParams['axes.labelsize'] = 13                   # X、Y轴标签字体大小
    mpl.rcParams['axes.titlesize'] = 20                   # 标题字体大小
    mpl.rcParams['lines.linewidth'] = 1.5                 # 线宽(默认1.5)
    mpl.rcParams['lines.markeredgewidth'] = 1             # 标记点附近线宽(默认1)
    mpl.rcParams['lines.markersize'] = 6                  # 标记点大小(默认6)
    # mpl.rcParams['agg.path.chunksize'] = 10000          # 解决绘图数据溢出报错
    
    plt.close('all')
    
    # 理想轨迹
    fig = plt.figure('理想轨迹')               # 创建绘图窗口
    plt.clf()                                 # 清除原图像
    ax = plt.axes(projection='3d')            # 创建3d绘图区域
    ax.grid()                                 # 生成网格
    
    ax.scatter(data.xi[0], data.zi[0], data.yi[0], s=30, c='r', marker='x', label='Start') # ax.scatter创建散点图
    ax.scatter(data.xi[-1], data.zi[-1], data.yi[-1], s=30, c='r', marker='o', label='End')# ax.scatter创建散点图
    ax.plot3D(data.xi, data.zi, data.yi, c='g', label='理想')                           # ax.plot3D创建三维线图

    ax.legend(loc='best').set_draggable(True) # 设置图例
    ax.set_xlabel('x (m)')                    # 设置坐标轴 
    ax.set_ylabel('z (m)')
    ax.set_zlabel('y (m)')
    ax.invert_xaxis()                         # 反转x轴(Z轴不能转???)
    ax.axis('auto')                           # 标度相同(mpl 3D 图无法设置equal)
    ax.set_title('理想轨迹')                   # 设置标题
    plt.show()                                # 显示图片

    # 轨迹跟踪控制
    fig = plt.figure('轨迹跟踪控制')           # 创建绘图窗口
    plt.clf()                                 # 清除原图像
    ax = plt.axes(projection='3d')            # 创建3d绘图区域
    ax.grid()                                 # 生成网格
    
    ax.scatter(data.xi[0], data.zi[0], data.yi[0], s=30, c='r', marker='x', label='Start') # ax.scatter创建散点图
    ax.scatter(data.xi[-1], data.zi[-1], data.yi[-1], s=30, c='r', marker='o', label='End')# ax.scatter创建散点图
    ax.plot3D(data.xi, data.zi, data.yi, c='g', label='理想', linestyle='-.')              # ax.plot3D创建三维线图
    ax.plot3D(data.xr, data.zr, data.yr, c='b', label='实际')                              # ax.plot3D创建三维线图

    ax.legend(loc='best').set_draggable(True) # 设置图例
    ax.set_xlabel('x (m)')                    # 设置坐标轴 
    ax.set_ylabel('z (m)')
    ax.set_zlabel('y (m)')
    ax.invert_xaxis()                         # 反转x轴(Z轴不能转???)
    ax.axis('auto')                           # 标度相同(mpl 3D 图无法设置equal)
    ax.set_title('轨迹跟踪控制')               # 设置标题
    plt.show()                                # 显示图片

    # 控制
    fig = plt.figure('控制量')      
    plt.clf()                                 
    plt.grid()                               
    plt.plot(data.ti, data.ui, label='理想', linestyle='-.') 
    plt.plot(data.tr, data.ur, label='实际')  
    plt.legend(loc='best').set_draggable(True) 
    plt.xlabel('t')                    
    plt.ylabel('u')
    plt.title('控制量')                     
    plt.show()                              

    # 速度
    fig = plt.figure('速度')               
    plt.clf()                                 
    plt.grid()                              
    plt.plot(data.ti, data.Vi, label='理想V', linestyle='-.') 
    plt.plot(data.tr, data.Vr, label='实际V')  
    plt.legend(loc='best').set_draggable(True)
    plt.xlabel('t')                       
    plt.ylabel('V (m/s)')
    plt.title('速度')              
    plt.show()                       

    # 角度
    fig = plt.figure('角度')               
    plt.clf()                                 
    plt.grid()                              
    plt.plot(data.ti, data.θi, label='理想θ', linestyle='-.') 
    plt.plot(data.ti, data.θr, label='实际θ') 
    plt.plot(data.tr, data.ψi, label='理想ψ', linestyle='-.') 
    plt.plot(data.tr, data.ψr, label='实际ψ')
    plt.legend(loc='best').set_draggable(True)
    plt.xlabel('t')                       
    plt.ylabel('angle (rad)')
    plt.title('角度')              
    plt.show()                 

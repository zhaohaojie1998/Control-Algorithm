"""ADRC 示例"""
import numpy as np
from controller.utils import timer, matplotlib_context
from controller.siso import ADRC, ADRCConfig


# 一维被控模型 (u 为 1 维, 跟踪第 1 个 y)
class PlantModel:
    def __init__(self, dt: float, with_noise=True):
        self.dt = dt
        self.t = 0                             # 初始时刻
        self.x = np.zeros(3, dtype=np.float32) # 初始状态
        self.u = 0                             # 初始控制
        self.with_noise = with_noise           # 是否存在干扰

    def __call__(self, u: np.ndarray):
        """更新状态和观测"""
        x_new = np.zeros_like(self.x)
        # 确保u是标量，即使控制器返回数组
        u = float(np.array(u).flatten()[0])
        if self.with_noise:
            f = -25 * self.x[1] + 33 * np.sin(np.pi*self.t) + 0.01*np.random.randn()
            x_new[0] = self.x[0] + self.x[1] * self.dt + 0.001*np.random.randn()
            x_new[1] = self.x[1] + self.x[2] * self.dt + 0.001*np.random.randn()
            x_new[2] = f + 133 * u
        else:
            f = -25 * self.x[1] + 33 * np.sin(np.pi*self.t)
            x_new[0] = self.x[0] + self.x[1] * self.dt
            x_new[1] = self.x[1] + self.x[2] * self.dt
            x_new[2] = f + 133 * u
        
        self.x = x_new
        self.u = u
        self.t += self.dt
        return self.y
    
    @property
    def y(self):
        """观测方程"""
        return self.x[0]


# 一维阶跃信号跟踪Demo        
@timer
def step_singnal_demo(cfg: ADRCConfig, with_noise=True):
    # 实例化控制算法
    dt = cfg.dt
    ctrl = cfg.build()
    print(ctrl)
    # 生成参考轨迹
    t_list = np.arange(0.0, 10.0, dt)
    v_list = np.sign(np.sin(t_list))
    # 初始化被控对象
    plant = PlantModel(dt, with_noise)
    y = plant.y
    # 仿真
    for i in range(len(t_list)):
        # 获取参考轨迹
        v = v_list[i]
        # 控制信号产生
        u = ctrl(y, v)
        # 更新观测
        y = plant(u)
    #end
    ctrl.show(name="Step")


# 一维余弦信号跟踪Demo
@timer
def cosine_singnal_demo(cfg: ADRCConfig, with_noise=True):
    # 实例化控制算法
    dt = cfg.dt
    ctrl = cfg.build()
    print(ctrl)
    # 生成参考轨迹
    t_list = np.arange(0.0, 10.0, dt)
    v_list = np.cos(t_list)
    # 初始化被控对象
    plant = PlantModel(dt, with_noise)
    y = plant.y
    # 仿真
    for i in range(len(t_list)):
        # 获取参考轨迹
        v = v_list[i]
        # 控制信号产生
        u = ctrl(y, v)
        # 更新观测
        y = plant(u)
    #end
    ctrl.show(name="Cosine")


# 状态调节器Demo
@timer
def state_regulator_demo(cfg: ADRCConfig, with_noise=True):
    # 实例化控制算法
    dt = cfg.dt
    ctrl = cfg.build()
    print(ctrl)
    # 初始化被控对象
    plant = PlantModel(dt, with_noise)
    y = plant.y
    # 仿真
    t_list = np.arange(0.0, 10.0, dt)
    for i in range(len(t_list)):
        # 控制信号产生
        u = ctrl(y)
        # 更新观测
        y = plant(u)
    #end
    ctrl.show(name="Regulator")



if __name__ == '__main__':
    cfg = ADRCConfig(
        dt = 0.001,
        dim = 1,
        # 跟踪微分器
        r = 100,                # 快速跟踪因子
        # 扩张状态观测器
        b0 = 133,               # 被控系统系数
        delta = 0.015,          # fal(e, alpha, delta)函数线性区间宽度
        eso_beta01 = 150,       # ESO反馈增益1
        eso_beta02 = 250,       # ESO反馈增益2
        eso_beta03 = 550,       # ESO反馈增益3
        # 非线性状态反馈控制率
        nlsef_beta1 = 10,       # 跟踪输入信号增益1
        nlsef_beta2 = 0.0009,   # 跟踪输入信号增益2
        nlsef_alpha1 = 200/201, # 0 < alpha1 < 1
        nlsef_alpha2 = 201/200, # alpha2 > 1 
    )
    with matplotlib_context():
        step_singnal_demo(cfg, with_noise=True)
    with matplotlib_context():
        cosine_singnal_demo(cfg, with_noise=True)

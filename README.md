<<<<<<< HEAD
# 控制算法

# Control Algorithm

### 控制器接口:

用于跟踪控制或反馈控制，即y信号(真实状态)跟踪v信号(理想状态)，控制器输入v和y，输出控制量u

![](Ctrl.png)

### 算法:

| 算法名           | 类名         | 洋名                                 | 备注                            |
| ---------------- | ------------ | ------------------------------------ | ------------------------------- |
| 位置式PID控制    | PID          | Proportion Integral Differential     | 自带抗积分饱和+积分分离功能     |
| 增量式PID控制    | IncrementPID | Increment PID Control                | 自带抗积分饱和+积分分离功能     |
| 自抗扰控制       | ADRC         | Active Disturbance Rejection Control |                                 |
| 线性二次型调节器 | LQR          | Linear Quadratic Regulator           | Flag                            |
| 启发算法控制     |              |                                      | Flag                            |
| 强化学习控制     |              |                                      | 需要pytorch，会降低CTRL的通用性 |

### 示例:

```python
import numpy as np
from PID import PID, PIDConfig
# 设置控制器
dim = 2 # 控制维度
cfg = PIDConfig(dt=0.1, dim=dim, Kp=[5,6]) # 调参
ctrl = PID(cfg) # 实例化控制器
# 生成输入信号
t_list = np.arange(0.0, 10.0, dt=cfg.dt)
v_list = np.ones((len(t_list), dim)) # 需要跟踪的信号 v: (dim, )
# 被控对象
def PlantModel(y, u, dt=cfg.dt):
    ...
    return y # y: (dim, ), u: (dim, )
# 仿真
y = np.zeros(2) # 初始状态
for v in v_list:
    u = ctrl(v, y) # 调用控制器
    y = PlantModel(y, u) # 更新被控信号
ctrl.show() # 输出仿真结果
```

### 仿真:

![](Result.png)

### **Requirement**:

python >= 3.9

numpy >= 1.22.3

matplotlib >= 3.5.1
=======
# Control-Algorithm
>>>>>>> a268140e0ed3f9ca86275589fd239e5e46f6781e

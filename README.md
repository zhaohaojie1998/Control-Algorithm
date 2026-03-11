# Control Algorithm

## 一、控制算法:

变量符号约定:
- v: 参考点或参考轨迹
- y: 系统输出（观测）
- x: 系统状态（当无法得到时需设计状态观测器）
- u: 控制量
- obs: RL环境观测, 对于跟踪问题，obs = [y, v]，对于调节问题，obs = [y]

### 1. 无模型

| 算法名 | 控制器类名 | 反馈类型 | 输入 | 输出 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 先进PID控制 <br /> Proportion Integral Differential | PID | 输出反馈 | y、v | u | 自带先进PID功能 |
| 增量式PID控制 <br /> Increment PID Control | IncrementPID | 输出反馈 | y、v | u | 自带先进PID功能 |
| 自抗扰控制 <br /> Active Disturbance Rejection Control | ADRC | 输出反馈 | y、v | u | 缺点：参数巨多。。。 |
| 线性自抗扰控制 <br /> Linear Active Disturbance Rejection Control | LADRC | 输出反馈 | y、v | u | 升级版ADRC, 参数较少 |

### 2. 基于模型

基于模型进行优化控制
- 对于使用状态反馈的控制器，需要额外设计状态观测器，要求系统能观。
- 对于使用输出反馈的控制器，控制器自带状态观测器，要求系统能观。
- 当系统不能控时，不保证控制效果。

#### 2.1 调节器

| 算法名 | 控制器类名 | 适用系统 | 能控 | 能观 | 反馈类型 | 输入 | 输出 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 线性二次型调节器 (状态调节版) <br /> Linear Quadratic Regulator | LQR (C=None) | 离散/连续LTI | √ | - | 状态反馈 | x | u |
| 线性二次型调节器 (输出调节版) <br /> Linear Quadratic Regulator | LQR | 离散/连续LTI | √ | √ | 状态反馈 | x | u |

#### 2.2 跟踪器

用于对y进行跟踪，需要跟踪x时，可将C设为单位矩阵。

| 算法名 | 控制器类名 | 适用系统 | 能控 | 能观 | 反馈类型 | 输入 | 输出 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 线性二次型积分 <br /> Linear Quadratic Integral | LQI | 离散/连续LTI | √ | √ | 状态反馈+输出反馈 | x、y、v | u |
| 线性二次型高斯 <br /> Linear Quadratic Gaussian | LQG | 离散/连续LTI | √ | √ | 输出反馈 | y、v | u |
| 时变线性二次型调节器 <br /> Time-Varying Linear Quadratic Regulator | TV_LQR | LTV |  |  | 状态反馈 | x、v | u |
| 迭代线性二次型调节器 <br /> Iterative Linear Quadratic Regulator | iLQR | NL |  |  | 状态反馈 | x、v | u |
| 模型预测控制 <br /> Model Predictive Control | MPC | 离散LTI |  |  | 状态反馈 | x、v_seq | u |
| 线性时变模型预测控制 <br /> Linear Time-Varying Model Predictive Control | LTV_MPC | 离散LTV |  |  | 状态反馈 | x、v_seq | u |
| 非线性模型预测控制 <br /> Nonlinear Model Predictive Control | NMPC | 离散NL |  |  | 状态反馈 | x、v_seq | u |

### 3. 智能控制 (基于仿真)

#### 3.1 强化学习

仿真环境用来产生训练数据
- RL通常不用观测器，可通过RNN、非对称Actor-Critic、蒸馏等手段直接从POMDP中学习输出反馈策略。
- RL算法仅用于训练模型，控制器类名统一为RLController，用于执行onnx推理。

| 算法名 | 类名 | 适用系统 | 反馈类型 | 输入 | 输出 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 软决策评估算法 <br /> Soft Actor-Critic | SAC | 离散/连续NL | 输出反馈 | obs | u | 收敛快，基本上不需要调参 |
| 近端策略优化算法 <br /> Proximal Policy Optimization | PPO | 离散/连续NL | 输出反馈 | obs | u | 收敛慢，但理论上稳的一批 |

#### 3.2 启发搜索（未实现）

仿真环境用来评估搜索解的好坏
- 原理类似MPC，直接搜索u_seq，u_seq带入环境模型评估搜索结果，利用启发算法优化，执行u_seq[0]，下一时刻重新搜索
（很多学阀喜欢用启发算法冒充AI骗经费，群体智能也是智能[狗头]）

| 算法名 | 控制器类名 | 适用系统 | 反馈类型 | 输入 | 输出 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 启发搜索算法 <br /> Heuristic Search | PSO等动物园算法 | 离散NL | 状态反馈 | x、v_seq | u | 暴力搜索就完事了，优化问题不存在了 <br /> 缺点：速度巨慢，基本上没实用价值 |

### 4. 模糊控制

| 算法名 | 控制器类名 | 反馈类型 | 输入 | 输出 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 模糊PID控制 <br /> Fuzzy PID Control | FuzzyPID | 输出反馈 | y、v | u | 模糊规则给PID调参 |

### 5. 状态估计器

暂未实现

## 二、控制器接口:

状态/输出跟踪器：
- 输入为 x | y 和 v，输出为 u

状态/输出调节器：
- 输入 x | y，v为None，输出为 u

![](图片/Ctrl.png)

#### 1. 控制器输入：

| 输入 | 定义 | 向量情况(ndarray) | 标量情况(ndarray/float) |
| --- | --- | --- | --- |
| v | 参考点 | shape = (dim_v, ) | shape = (1, ) / float |
| v_seq | 参考轨迹片段 | shape = (n, dim, ) | shape = (n, ) |
| y | 系统输出（观测） | shape = (dim_y, ) | shape = (1, ) / float |
| x | 系统状态 | shape = (dim_x, ) | shape = (1, ) / float |

#### 2. 控制器输出：

当前时刻的控制量u，形状为(dim_u, )的向量（一维ndarray），无论x、y、v是否为标量，输出u都是向量，即使dim_u=1时也不输出float

#### 3. 控制器参数：

控制器参数设置成float时，将自动广播成向量或者对角矩阵

对于SISO控制器，超参数为float；当dim_u>1时，也可设置为向量（设置成一维list或ndarray），为每个u的维度单独调参

对于MIMO控制器，超参为矩阵 (设置成二维list或ndarray)

对于AI控制器，超参为AI算法超参(float)，和控制无关

## 三、用法示例:

### 1. SISO控制器示例：

```python
import numpy as np
from controller.siso import PIDConfig, PID
from controller.utils import matplotlib_context

# 设置控制器
dim = 2  # SISO需要指定信号维度, 每个维度控制器独立
cfg = PIDConfig(dt=0.1, dim=dim, Kp=[5, 6], Ki=0.1, Kd=1)  # 调参
pid = cfg.build()  # 实例化控制器
# or: pid = PID(cfg)
print(pid)  # 打印控制器参数

# 生成输入信号
t_list = np.arange(0.0, 10.0, cfg.dt)
v_list = np.ones((len(t_list), dim))  # 需要跟踪的信号 v: (dim, )

# 被控对象
def dynamics(y, u, dt=cfg.dt):
    ...
    return y

# 仿真
y = np.zeros(2)  # 被控信号初值
for v in v_list:
    u = pid(y, v)  # 第一个为实际值, 第二个为参考值（调节器为None）
    y = dynamics(y, u)  # 更新被控信号

# 绘图输出
with matplotlib_context():
    pid.show(save_img=True)
```

### 2. MIMO控制器示例：

```python
import numpy as np
from controller import LTISystem, LQR
from controller.utils import matplotlib_context

# 线性系统
A = np.array([[0, 1], [0, 0]])
B = np.array([[0], [1]])
sys = LTISystem(A, B, Ts=None) # Ts=None为连续系统, C=None没有观测方程
# 可通过以下方法检查线性系统性质
print(sys.is_controllable()) # 能控性
print(sys.is_stabilizable()) # 镇定性
print(sys.is_observable())   # 能观性, C=None返回None
print(sys.is_detectable())   # 可检测性, C=None返回None
print(sys.is_stable())       # 稳定性
print(sys.is_lyapunov_stable(np.eye(2))) # 是否Lyapunov稳定

step_dt = 0.01
max_steps = 1000

# 设置控制器，MIMO根据模型自动推断维度
Q = 2  # 设置为float时，自动广播成对角矩阵
R = 0.1
lqr = LQR(sys, Q, R, dt=step_dt) # C=None时为状态调节器, 否则为输出调节器
print(lqr)  # 打印控制器参数

# 仿真
x = np.zeros(2)  # 被控信号初值
for _ in range(max_steps):
    u = lqr(x)  # 第一个为实际值, 第二个为参考值（调节器为None）
    dx = A @ x + B @ u  # 更新被控信号
    x += dx * step_dt

# 绘图输出
with matplotlib_context():
    lqr.show(save_img=True)
```

### 3. RL控制器示例：

```python
import gymnasium as gym
from controller.ai.rl import PPO, RLController
from controller.utils import matplotlib_context

# 实例化仿真环境
env = gym.make("Pendulum-v1")
print(env.observation_space)
print(env.action_space)

# 训练策略模型 (已有onnx模型可省略)
ppo = PPO(env, gamma=0.99, lr_actor=0.0003, lr_critic=0.0003, clip_range=0.2)
print(ppo)
ppo.train(max_env_steps=100000) # 训练
ppo.save_onnx("ppo_pendulum.onnx") # 导出onnx控制模型

# 实例化RL控制器
rl_ctrl = RLController("ppo_pendulum.onnx", dt=0.05)
# or: rl_ctrl = ppo.get_controller("ppo_pendulum.onnx", dt=0.05)
print(rl_ctrl)

# 仿真
env = gym.make("Pendulum-v1", render_mode="human")
obs, _ = env.reset()
rl_ctrl.reset()
for _ in range(max_env_steps):
    u = rl_ctrl(obs)
    obs, _, terminated, truncated, _ = env.step(u)
    if terminated or truncated:
        with matplotlib_context():
            rl_ctrl.show(save_img=True)
        obs, _ = env.reset()
        rl_ctrl.reset()
```


## 四、控制器效果图:

#### 1. PID控制算法：

参数忘了。。。

![](图片/Result0.png)

#### 2. ADRC控制算法：

![](图片/Result1.png)

#### 3. 模糊PID控制算法：

两组对比图参数分别为Kp=5,Ki=0,Kd=0.2和Kp=5,Ki=0,Kd=0.1

![img](图片/Fuzzy.png)

#### 4. LQR控制器：

##### LQR状态调节器

![](图片/LQR_StateRegulator0.png)

![](图片/LQR_StateRegulator1.png)

##### LQR输出调节器（基于状态反馈控制，需要设计状态观测器）

![](图片/LQR_OutputRegulator0.png)

示例使用带噪声的状态假装观测结果，因此u有抖动，实际u的好坏取决于观测器好坏

![](图片/LQR_OutputRegulator1.png)

##### LQI输出跟踪器（基于状态反馈控制，需要设计状态观测器）

![](图片/LQI1.png)

![](图片/LQI2.png)

## 五、小车位置跟踪控制:

![](图片/PID0.png)

## 六、无人机位置跟踪控制：

![](图片/ADRC0.png)

![](图片/ADRC1.png)

![](图片/ADRC2.png)

![](图片/ADRC3.png)

![](图片/ADRC4.png)

![](图片/ADRC5.png)

![](图片/ADRC6.png)
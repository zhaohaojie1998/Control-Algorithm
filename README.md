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

### 2. 基于模型

基于模型进行优化控制
- 对于使用状态反馈的控制器，需要设计状态观测器

| 算法名 | 控制器类名 | 反馈类型 | 输入 | 输出 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 线性二次型调节器 (状态调节版) <br /> Linear Quadratic Regulator | LQR_StateRegulator | 状态反馈 | x | u | 线性时不变系统，要求能控 |
| 线性二次型调节器 (输出调节版) <br /> Linear Quadratic Regulator | LQR_OutputRegulator | 状态反馈 | x | u | 线性时不变系统，要求能控能观 |
| 线性二次型调节器 (输出跟踪版) <br /> Linear Quadratic Regulator | LQR_OutputTracker | 状态反馈 | x、v | u | 线性时不变系统，要求能控能观 |
| 时变线性二次型调节器 <br /> Time-Varying Linear Quadratic Regulator | TVLQR | 状态反馈 | x、v | u | 线性时变系统 |
| 迭代线性二次型调节器 <br /> Iterative Linear Quadratic Regulator | iLQR | 状态反馈 | x、v | u | 非线性系统 |
| 线性二次型积分 (状态调节版) <br /> Linear Quadratic Integral | LQI_StateRegulator | 状态反馈+输出反馈 | x、y | u | 带积分器的LQR |
| 线性二次型高斯 (状态调节版) <br /> Linear Quadratic Gaussian | LQG_StateRegulator | 输出反馈 | y | u | 带卡尔曼滤波的LQR |
| 模型预测控制 <br /> Model Predictive Control | MPC | 状态反馈 | x、v_seq | u | 非线性系统<br />缺点：计算慢，且需要知道未来n步的v |

### 3. 智能控制 (基于仿真)

#### 3.1 强化学习 (未实现)

仿真环境用来产生训练数据
- RL通常不用观测器，可通过RNN、非对称Actor-Critic、蒸馏等手段直接从POMDP中学习输出反馈策略

| 算法名 | 类名 | 反馈类型 | 输入 | 输出 | 备注 |
| --- | --- | --- | --- | --- | --- |
| SAC算法 <br /> Soft Actor-Critic | SAC | 输出反馈 | obs | u | 收敛快 |
| PPO算法 <br /> Proximal Policy Optimization | PPO | 输出反馈 | obs | u | 收敛慢，但稳的一批 |

#### 3.2 启发搜索（未实现）

仿真环境用来评估搜索解的好坏
- 原理类似MPC，直接搜索u_seq，u_seq带入环境模型评估搜索结果，利用启发算法优化，执行u_seq[0]，下一时刻重新搜索
（很多学阀喜欢用启发算法冒充AI骗经费，群体智能也是智能[狗头]）

| 算法名 | 控制器类名 | 反馈类型 | 输入 | 输出 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 启发搜索算法 <br /> Heuristic Search | PSO等动物园算法 | 状态反馈 | x、v_seq | u | 缺点：速度巨慢，基本上没实用价值 |

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
t_list = np.arange(0.0, 10.0, dt=cfg.dt)
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
from controller.mimo import LQR_StateRegulator
from controller.utils import matplotlib_context

# 线性系统
A = np.array([[0, 1], [0, 0]])
B = np.array([[0], [1]])
dt = 0.01
max_steps = 1000

# 设置控制器，MIMO根据模型自动推断维度
Q = 2  # 设置为float时，自动广播成对角矩阵
R = 0.1
lqr = LQR_StateRegulator(A, B, Q, R, dt=dt, discrete=True)
print(lqr)  # 打印控制器参数

# 仿真
x = np.zeros(2)  # 被控信号初值
for _ in range(max_steps):
    u = lqr(x)  # 第一个为实际值, 第二个为参考值（调节器为None）
    x = A @ x + B @ u  # 更新被控信号

# 绘图输出
with matplotlib_context():
    lqr.show(save_img=True)
```

### 3. RL控制器示例：

```python
import gymnasium as gym
from controller.ai.rl import PPO, RLController

# 实例化仿真环境
env = gym.make("Pendulum-v1")
print(env.observation_space)
print(env.action_space)

# 训练策略模型 (已有onnx模型可省略)
ppo = PPO(env, gamma=0.99, lr=0.0003, clip_range=0.2)
print(ppo)
ppo.train(max_steps=100000) # 训练
ppo.save_onnx("ppo_pendulum.onnx") # 导出onnx控制模型

# 实例化RL控制器
rl_ctrl = RLController("ppo_pendulum.onnx")
# or: rl_ctrl = ppo.get_controller("ppo_pendulum.onnx")

# 仿真
env = gym.make("Pendulum-v1", render_mode="human")
obs, _ = env.reset()
rl_ctrl.reset()
for _ in range(max_steps):
    u = rl_ctrl(obs)
    obs, _, terminated, truncated, _ = env.step(u)
    if terminated or truncated:
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
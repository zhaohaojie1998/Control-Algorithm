"""强化学习控制器"""
from .rl_controller import RLController

# RL算法
try:
    from .sac import SAC
except ImportError:
    def SAC(*args, **kwargs):
        raise ImportError("please run 'pip install torch onnx gymnasium tensorboard' to use SAC.")

try:
    from .ppo import PPO
except ImportError:
    def PPO(*args, **kwargs):
        raise ImportError("please run 'pip install torch onnx gymnasium tensorboard' to use PPO.")

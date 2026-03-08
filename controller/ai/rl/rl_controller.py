# -*- coding: utf-8 -*-
"""
强化学习控制器 (非线性系统)

@author: https://github.com/zhaohaojie1998
"""

''' RL '''
# model free controller
import numpy as np

from ...base import BaseController
from ...types import SignalLike, NdArray

__all__ = [
    "RLController"
]


class RLController(BaseController):
    """强化学习控制器"""

    def __init__(self, onnx_path: str, dt: float):
        """
        Args:
            onnx_path (str): onnx模型路径
            dt (float): 控制器步长
        """
        super().__init__()
        self.dt = dt
        self.t = 0.0

        # 加载模型
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("please run 'pip install onnxruntime' to use RLController.") from e
        self.onnx_path = onnx_path
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [i.name for i in self.session.get_outputs()]
        
        # RNN处理
        self.rnn_type = self._get_rnn_type()
        self.hidden_state = self._init_hidden()

        # 绘图数据
        self.logger.obs = []

    def _get_rnn_type(self) -> str:
        has_h = "h_in" in self.input_names
        has_c = "c_in" in self.input_names
        if not has_h and not has_c:
            return ""
        return "LSTM" if has_c else "GRU"
    
    def _init_hidden(self) -> dict[str, np.ndarray]:
        hidden = {}
        if not self.rnn_type:
            return hidden

        # 初始化 h_in
        h_info = next(i for i in self.session.get_inputs() if i.name == "h_in")
        h_shape = [1 if d == -1 else d for d in h_info.shape]
        hidden["h_in"] = np.zeros(h_shape, dtype=np.float32)
        
        # 初始化 c_in
        if self.rnn_type == "LSTM":
            c_info = next(i for i in self.session.get_inputs() if i.name == "c_in")
            c_shape = [1 if d == -1 else d for d in c_info.shape]
            hidden["c_in"] = np.zeros(c_shape, dtype=np.float32)
        return hidden

    def reset(self):
        super().reset()
        self.t = 0.0
        # 重置RNN状态
        self.hidden_state = self._init_hidden()

    def __call__(self, obs: SignalLike) -> NdArray:
        """
        Args:
            obs (SignalLike): 环境观测

        Returns:
            u (NdArray): RL控制量
        """
        obs = np.array(obs).reshape(1, -1).astype(np.float32) # (batch_size, obs_dim)

        inputs = {"obs": obs}
        if self.rnn_type:
            inputs.update(self.hidden_state)
        
        outputs = self.session.run(None, inputs)
        u = outputs[0].ravel() # (act_dim,)
        
        # 更新RNN隐藏状态
        if self.rnn_type == "GRU":
            self.hidden_state["h_in"] = outputs[1]
        elif self.rnn_type == "LSTM":
            self.hidden_state["h_in"] = outputs[1]
            self.hidden_state["c_in"] = outputs[2]

        # 绘图数据记录
        self.t += self.dt
        self.logger.t.append(self.t)
        self.logger.obs.append(obs.ravel())
        self.logger.u.append(u)
        return u
    
    def show(self, name='', save_img=False):
        super().show(name=name, save_img=save_img)
        self._add_figure(name=name, title='Observation', t=self.logger.t,
                        y1=self.logger.obs, y1_label='obs',
                        xlabel='time', ylabel='obs', save_img=save_img)
    
    def __repr__(self):
        rnn_info = f", rnn_type={self.rnn_type}" if self.rnn_type else ""
        return f"RL Controller (dt={self.dt}, model={self.onnx_path}{rnn_info})"
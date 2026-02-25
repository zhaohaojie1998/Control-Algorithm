# -*- coding: utf-8 -*-
"""
类型定义文件
Created on Sat Jul 23 23:59:33 2022

@author: https://github.com/zhaohaojie1998
"""

from typing import Union
import numpy as np

ListLike = Union[list, np.ndarray]
"""绘图数据"""

SignalLike = Union[float, list, np.ndarray]
"""输入信号/超参, 标量或向量"""

ScalarLike = Union[int, float]
"""标量数据类型"""

NdArray = np.ndarray
"""控制信号, 向量"""

MatLike = Union[list[list[float]], np.ndarray]
"""矩阵数据类型"""

# -*- coding: utf-8 -*-
"""
实用工具
Created on Mon Mar 13 2023 16:06:44
 
@auther: https://github.com/zhaohaojie1998
"""
import os
import time
import random
import functools
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'setup_seed',
    'matplotlib_context',
    'timer',
    'tic', 'toc',
]


# 随机种子设置
def setup_seed(seed=None):
    '''为torch, torch.cuda, numpy, random设置随机种子'''
    if seed is None:
        return
    try:
        import torch
        torch.manual_seed(seed)            # 为CPU设置随机种子
        torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    finally:
        np.random.seed(seed)               # 为numpy设置随机种子
        random.seed(seed)                  # 为random设置随机种子
        os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化


# Matplotlib上下文管理器
@contextmanager
def matplotlib_context():
    '''# Matplotlib绘图上下文管理器\n
    进入时关闭所有已打开的窗口, 并修复中文乱码, 退出时弹出当前所有的绘图窗口：\n
    >>> with matplotlib_context():
    >>>     ctrl_pos.show(name="位置环")
    >>>     ctrl_vel.show(name="速度环")
    >>>     ctrl_att.show(name="姿态环")
    >>>     plt.figure("轨迹图")
    >>>     plt.plot(your_data) # 其余绘图
    >>>     plt.title("轨迹图")
    '''
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 修复字体bug
        plt.rcParams['axes.unicode_minus'] = False              # 用来正常显示负号
        # plt.rcParams['agg.path.chunksize'] = 10000            # 解决绘图数据溢出报错
        plt.close('all')
        yield

    finally:
        plt.show()


# 函数计时器
def timer(func=None, /, *, name='', CN=True, digit=6):
    '''函数计时器\n
    >>> @timer
    >>> def func1(...):
    >>>     ...
    >>> @timer(name='MyFunc', digit=8)
    >>> def func2(...):
    >>>     ...
    '''
    if func is not None and not name:
        name = func.__name__
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            res = func(*args, **kwargs)
            stop_time = time.perf_counter()
            if CN: # matlab 风格
                info = name + ' 历时' if name else '历时'
                print('%s %f 秒。' %(info, round(stop_time - start_time, digit)))
            else:
                info = name + ' elapsed time is' if name else 'Elapsed time is'
                print('%s %f seconds.' %(info, round(stop_time - start_time, digit)))
            return res
        return wrapper
    
    if func is None:
        return decorator # @run_time(), 返回一个装饰器
        # func = run_time(param=...)(func)
    else:
        return decorator(func) # @run_time, 返回被装饰的函数
        # func = run_time(func)


# matlab计时器
def tic(): 
    '''计时开始'''
    if 'global_tic_time' not in globals():
        global global_tic_time
        global_tic_time = []
    global_tic_time.append(time.perf_counter())
    
def toc(name='', *, CN=True, digit=6): 
    '''计时结束'''
    if 'global_tic_time' not in globals() or not global_tic_time: # 未设置全局变量或全局变量为[]
        print('未设置tic' if CN else 'tic not set')  
        return
    elapsed = time.perf_counter() - global_tic_time.pop()   
    if CN: # matlab 风格
        print('%s历时 %f 秒。' % (name, round(elapsed, digit)))
    else:
        info = name + ' elapsed time is' if name else 'Elapsed time is'
        print('%s %f seconds.' % (info, round(elapsed, digit)))
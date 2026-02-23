# -*- coding: utf-8 -*-
"""
实用工具
Created on Mon Mar 13 2023 16:06:44
 
@auther: https://github.com/zhaohaojie1998
"""
#
''' PYTHON工具 '''
import os
import time
import random
import platform
import numpy as np
from datetime import datetime
from functools import wraps

__all__ = [
    'get_str_time',
    'tic',
    'toc',
    'TicToc',
    'run_time_wraps',
    'setup_seed',
]


# 获取当前时间
def get_str_time(mode=0):
    '''获取当前时间字符串
    mode=0 : 20230421-133055
    mode=1 : 2023-04-21 13-30-55
    mode=any : Dec10_20-22-30_YOGA14s
    '''
    if mode == 0:
        return datetime.now().strftime("%Y%m%d-%H%M%S")                # 20230421-133055
    elif mode == 1:
        return datetime.now().strftime("%Y-%m-%d %H-%M-%S")            # 2023-04-21 13-30-55
    return datetime.now().strftime("%b%d_%H-%M-%S_") + platform.node() # Dec10_20-22-30_YOGA14s


# matlab计时器
def tic(): 
    '''计时开始'''
    if 'global_tic_time' not in globals():
        global global_tic_time
        global_tic_time = []
    global_tic_time.append(time.time())
    
def toc(name='', *, CN=True, digit=6): 
    '''计时结束'''
    if 'global_tic_time' not in globals() or not global_tic_time: # 未设置全局变量或全局变量为[]
        print('未设置tic' if CN else 'tic not set')  
        return
    name = name+' ' if (name and not CN) else name
    if CN:
        print('%s历时 %f 秒。' % (name, round(time.time() - global_tic_time.pop(), digit)))
    else:
        print('%sElapsed time is %f seconds.' % (name, round(time.time() - global_tic_time.pop(), digit)))


# 函数计时装饰器
def run_time_wraps(func=None, /, *, name='', CN=True, digit=6):
    '''函数计时器\n
    >>> @run_time_wraps # or @run_time_wraps(kwarg=...)
    >>> def func(...):
    >>>     ...
    '''
    name = name+' ' if (name and not CN) else name
  
    def decorator(func):
        @wraps(func) # 装饰器不修改原函数信息
        def wrapper(*args, **kwargs):
            start_time = time.time()
            res = func(*args, **kwargs)
            stop_time = time.time()
            if CN:
                print('%s历时 %f 秒。' %(name, round(stop_time - start_time, digit)))
            else:
                print('%sElapsed time is %f seconds.' %(name, round(stop_time - start_time, digit)))
            return res
        return wrapper # 返回被装饰的函数
    
    if func is None:
        return decorator # @run_time(), 返回一个装饰器
        # func = run_time(param=...)(func)
    else:
        return decorator(func) # @run_time, 返回被装饰的函数
        # func = run_time(func)


# 多功能计时器
class TicToc:
    '''# 多功能计时器\n
    模式0 - 上下文管理:
    >>> In [0]: with TicToc(name='MyCode'):
    >>>             code block
    >>> Out[0]: MyCode历时 0.000001 秒。\n
    模式1 - 函数装饰器:
    >>> In [1]: @TicToc(name='MyFunc', digit=3)
    >>>         def func1():
    >>>             ...
    >>>         @TicToc # 不推荐该用法, 可改成 @TicToc()
    >>>         def func2():
    >>>             ...
    >>>         func1() # func1为被装饰的func1
    >>>         func2() # func2为TicToc的实例对象
    >>> Out[1]: MyFunc历时 0.001 秒。
    >>>         历时 0.000001 秒。\n
    模式2 - Matlab工具:
    >>> In [2]: t = TicToc(CN=False)
    >>>         t.tic
    >>>         time.sleep(1)
    >>>         if 1:
    >>>             t.tic
    >>>             time.sleep(2)
    >>>             t.toc
    >>>         t.toc # or: total_time = t.toc
    >>> Out[2]: Elapsed time is 2.000003 seconds.
    >>>         Elapsed time is 3.000005 seconds.
    '''

    def __init__(self, func=None, /, *, name='', CN=True, digit=6):
        self.func = func
        self.name = name+' ' if (name and not CN) else name
        self.CN = CN
        self.digit = digit
        self.__tic_time = []

    # with上下文管理 -> with TicToc(): 
    def __enter__(self):
        self.tic
    def __exit__(self, type, value, traceback):
        self.toc

    # 函数装饰器 -> @TicToc, @TicToc()
    def __call__(self, *args, **kwargs):
        # @TicToc() -> 返回被装饰的函数对象 -> func = TicToc()(func)
        if self.func is None:
            func = args[0] # 此时func为arg里的第一个参数
            @wraps(func) # 装饰器不修改原函数信息
            def wrapper(*params, **kw_params):
                self.tic
                res = func(*params, **kw_params)
                self.toc
                return res
            return wrapper
        # @TicToc -> 返回TicToc的可调用实例对象 -> func = TicToc(func), func(...) == TicToc_obj(...)
        else:
            self.tic
            res = self.func(*args, **kwargs)
            self.toc
            return res

    # matlab计时器 -> T.tic, T.toc
    @property
    def tic(self):
        self.__tic_time.append(time.time())
    
    @property
    def toc(self) -> float:
        if not self.__tic_time:
            print('未设置tic' if self.CN else 'tic not set')  
            return
        elapsed = time.time() - self.__tic_time.pop()   
        if self.CN:
            print(f'{self.name}历时 {round(elapsed, self.digit)} 秒。')
        else:
            print(f'{self.name}Elapsed time is {round(elapsed, self.digit)} seconds.')
        return elapsed


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
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 23:59:33 2022

@author: HJ
"""

''' 控制器 '''
from abc import ABC, abstractmethod
from utils import get_path, tic, toc
import pylab as pl


class BaseController(ABC, object):
    def __init__(self):
        # common参数
        self.name = 'Controller'
        self.dt = 0.001
        self.dim = 1     # 跟踪信号维度
        
        # common存储器
        self.list_t= []  # 时间
        self.list_u = [] # 控制
        self.list_y = [] # 实际信号
        self.list_v = [] # 输入信号
        
    @abstractmethod
    def __call__(self):
        pass
    
    def __repr__(self):
        return self.name +' Controller'
    
    @abstractmethod
    def show(self):
        pass
    
    def basic_plot(self, save = False):
        # 绘图配置
        pl.mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 修复字体bug
        pl.mpl.rcParams['axes.unicode_minus'] = False            # 用来正常显示负号
        pl.mpl.rcParams['figure.figsize'] =  [10, 8]             # 视图窗口大小，英寸表示(默认[6.4,4.8])
        pl.mpl.rcParams['font.size'] = 20                        # 所有字体大小(默认12)
        pl.mpl.rcParams['xtick.labelsize'] = 20                  # X轴标签字体大小
        pl.mpl.rcParams['ytick.labelsize'] = 20                  # Y轴标签字体大小
        pl.mpl.rcParams['axes.labelsize'] = 20                   # X、Y轴刻度标签字体大小
        pl.mpl.rcParams['axes.titlesize'] = 20                   # 坐标轴字体大小
        pl.mpl.rcParams['lines.linewidth'] = 1.5                 # 线宽(默认1.5)
        pl.mpl.rcParams['lines.markeredgewidth'] = 1             # 标记点附近线宽(默认1)
        pl.mpl.rcParams['lines.markersize'] = 6                  # 标记点大小(默认6)
        # pl.mpl.rcParams['agg.path.chunksize'] = 10000          # 解决绘图数据溢出报错
        pl.close('all')                                          # 关闭所有窗口
        
        # 响应曲线
        self._figure(fig_name='Response Curve', t=self.list_t,
                     y1=self.list_y, y1_label='Real Signal',
                     y2=self.list_v, y2_label='Input Signal',
                     xlabel='time', ylabel='response signal', save=save)
        # 控制曲线
        self._figure(fig_name='Control Law', t=self.list_t,
                     y1=self.list_u, y1_label='Control Signal',
                     xlabel='time', ylabel='control signal', save=save)
        
            
    def _figure(self, fig_name, t, y1, y1_label, y2=None, y2_label=None, xlabel='time', ylabel='signal', save=False):
        # 图例添加
        def get_label(label):
            if self.dim != 1:
                lb = []
                for i in range(self.dim):
                    lb.append(label+' {}'.format(i+1))
            else:
                lb = label
            return lb
        lb1 = get_label(y1_label)
        lb2 = get_label(y2_label) if y2_label is not None else y2_label

        # 绘图
        pl.figure(self.name + ' ' + fig_name, (10,8)) # 创建绘图窗口
        pl.clf() # 清除原图像
        pl.plot(t, y1, label=lb1) 
        if y2 is not None:
            pl.plot(t, y2, label=lb2, linestyle='-.') 
        pl.xlabel(xlabel, fontsize = 20) # x轴标签
        pl.ylabel(ylabel, fontsize = 20) # y轴标签
        pl.xticks(fontsize = 20) # x轴刻度设置
        pl.yticks(fontsize = 20) # y轴刻度设置
        pl.grid() # 生成网格
        pl.legend(loc='best', fontsize = 20).set_draggable(True) # 显示图例
        pl.title(self.name + ' ' + fig_name, fontsize = 20)      # 标题
        if save:
            path = get_path(self.name + ' ' + fig_name + ' .png', self.name)
            pl.savefig(path)
            
            
    # 无人机轨迹跟踪控制
    def _figure3D(self, fig_name, save=False):
        if self.dim != 3:
            return
        v = pl.array(self.list_v) # 参考轨迹
        y = pl.array(self.list_y) # 实际轨迹
        x1 = v[:,0]; y1 = v[:,1]; z1 = v[:,2]
        x2 = y[:,0]; y2 = y[:,1]; z2 = y[:,2]
        
        fig = pl.figure(self.name + ' ' + fig_name, (10,8)) # 创建绘图窗口
        pl.clf()                              # 清除原图像       
        ax = fig.add_subplot(projection='3d') # 创建3D绘图
        ax.grid()                             # 生成网格
        ax.axis('auto')
        ax.plot(x1,y1,z1,'r',label='Ideal Trajectory')
        ax.plot(x2,y2,z2,'b',label='Real Trajectory')
        ax.set_xlabel('x')                    # x轴标签
        ax.set_ylabel('y')                    # y轴标签
        ax.set_zlabel('z')                    # z轴标签
        ax.legend(loc='best').set_draggable(True) # 显示图例
        pl.title(self.name + ' ' + fig_name)      # 标题
        if save:
            path = get_path(self.name + ' ' + fig_name + ' .png', self.name)
            pl.savefig(path)

          




    
            
''' 一维信号跟踪demo '''          
def demo0(algo, cfg):
    # 实例化控制算法
    dt = cfg.dt
    ctrl = algo(cfg)
    print('算法：', ctrl)
    
    # 生成参考轨迹
    t_list = pl.arange(0.0, 10.0, dt)
    v_list = pl.sign(pl.sin(t_list))
    
    # 定义动力学模型
    def PlantModel(y, u, t, dt):
        y1 = pl.zeros(3)
        f = -25 * y[1] + 33 * pl.sin(pl.pi*t) + 0.01*pl.randn(1)
        y1[0] = y[0] + y[1] * dt + 0.001*pl.randn(1)
        y1[1] = y[1] + y[2] * dt + 0.001*pl.randn(1)
        y1[2] = f + 133 * u
        return y1
    
    # 初始化状态
    y3 = pl.zeros(3)
    
    # 仿真
    tic()
    for i in range(len(t_list)):
        # 获取参考轨迹
        t = t_list[i]
        v = v_list[i]
        # 控制信号产生
        u = ctrl(v, y3[0])
        # 动力学环节
        y3 = PlantModel(y3, u, t, dt)

    toc()
    
    # 绘图
    ctrl.show(save = False)
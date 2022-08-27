# 工具包 #

''' MATLAB计时器 '''
from datetime import datetime
# 计时开始
def tic():
    global tic_
    tic_ = datetime.now()
# 计时结束
def toc():
    toc_ = datetime.now()
    print('Elapsed time is %f seconds' % (toc_-tic_).total_seconds())
    
    
''' 路径管理 '''
import sys, os
current_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(current_path) # 父路径
#sys.path.append(parent_path) # 添加路径到系统路径sys.path
#sys.path.append(current_path) # 添加路径到系统路径sys.path

def get_path(file_name, algo_name):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = os.path.join('.', 'figure', algo_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path) # 递归创建目录 .\figure\algo
        
    file_name = os.path.splitext(file_name)[0] + current_time \
                + os.path.splitext(file_name)[-1]
    file_path = os.path.join(file_path, file_name)
    return file_path

    
  
# ''' 设置随机数种子 '''
# import os
# import torch
# import numpy as np
# import random
# def setup_seed(seed):
#      torch.manual_seed(seed)            # 为CPU设置随机种子
#      torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
#      torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
#      torch.backends.cudnn.deterministic = True
#      #torch.backends.cudnn.benchmark = False
#      #torch.backends.cudnn.enabled = False
#      np.random.seed(seed)               # 为numpy设置随机种子
#      random.seed(seed)                  # 为random设置随机种子
#      os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现。
     
# A simple torch style logger
# (C) Wei YANG 2017
from __future__ import absolute_import
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

__all__ = ['Logger', 'LoggerMonitor', 'savefig']

def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)
    
def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + '(' + name + ')' for name in names]

# 保留原始注释掉的Logger类作为参考
# class Logger(object):
#     '''Save training process to log file with simple plot function.'''
#     def __init__(self, fpath, title=None, resume=False): 
#         self.file = None
#         self.resume = resume
#         self.title = '' if title == None else title
#         if fpath is not None:
#             if resume: 
#                 self.file = open(fpath, 'r') 
#                 name = self.file.readline()
#                 self.names = name.rstrip().split('\t')
#                 self.numbers = {}
#                 for _, name in enumerate(self.names):
#                     self.numbers[name] = []

#                 for numbers in self.file:
#                     numbers = numbers.rstrip().split('\t')
#                     for i in range(0, len(numbers)):
#                         self.numbers[self.names[i]].append(numbers[i])
#                 self.file.close()
#                 self.file = open(fpath, 'a')  
#             else:
#                 self.file = open(fpath, 'w')

#     def set_names(self, names):
#         if self.resume: 
#             pass
#         # initialize numbers as empty list
#         self.numbers = {}
#         self.names = names
#         for _, name in enumerate(self.names):
#             self.file.write(name)
#             self.file.write('\t')
#             self.numbers[name] = []
#         self.file.write('\n')
#         self.file.flush()


#     def append(self, numbers):
#         assert len(self.names) == len(numbers), 'Numbers do not match names'
#         for index, num in enumerate(numbers):
#             self.file.write("{0:.6f}".format(num))
#             self.file.write('\t')
#             self.numbers[self.names[index]].append(num)
#         self.file.write('\n')
#         self.file.flush()

#     def plot(self, names=None):   
#         names = self.names if names == None else names
#         numbers = self.numbers
#         for _, name in enumerate(names):
#             x = np.arange(len(numbers[name]))
#             plt.plot(x, np.asarray(numbers[name]))
#         plt.legend([self.title + '(' + name + ')' for name in names])
#         plt.grid(True)

#     def close(self):
#         if self.file is not None:
#             self.file.close()

class Logger(object):
    # 新增mode参数，支持'w'（覆盖）和'a'（追加）模式，默认为'w'
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout  # 确保始终初始化console属性
        self.file = None
        self.fpath = fpath
        
        if fpath is not None:
            # 确保日志目录存在
            log_dir = os.path.dirname(fpath)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # 根据mode参数打开文件（支持续训时追加日志）
            self.file = open(fpath, mode, encoding='utf-8')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        # 同时输出到控制台和文件
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        # 刷新缓冲区
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        # 只关闭文件句柄，不关闭控制台（避免后续print失效）
        if self.file is not None:
            self.file.close()
            self.file = None


class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)

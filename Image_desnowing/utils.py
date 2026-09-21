import time
import numpy as np


class Adder(object):
    """简易累加器,用于统计一段时间的累计数值并求平均(例如损失/PSNR 的滑动平均)。"""
    def __init__(self):
        self.count = 0       # 累计次数
        self.num = float(0)  # 累计的数值总和

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        """每次调用(例如 adder(loss))就把 num 累加进去,并记录次数。"""
        self.count += 1
        self.num += num

    def average(self):
        """返回到目前为止所有数值的平均值。"""
        return self.num / self.count


class Timer(object):
    """计时器,option 决定返回时间的单位:'s' 秒 / 'm' 分钟 / 其余为小时。"""
    def __init__(self, option='s'):
        self.tm = 0                   # 记录 tic 时刻的时间戳
        self.option = option
        if option == 's':
            self.devider = 1          # 秒
        elif option == 'm':
            self.devider = 60         # 分钟
        else:
            self.devider = 3600       # 小时

    def tic(self):
        """开始计时。"""
        self.tm = time.time()

    def toc(self):
        """返回从 tic 到现在经过的时间(按设置的单位换算)。"""
        return (time.time() - self.tm) / self.devider


def check_lr(optimizer):
    """从优化器中取回当前学习率。通常用于打印/日志监控学习率衰减。"""
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr
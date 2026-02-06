import time
import numpy as np
from collections import deque
__all__ = ['Timer']


class Timer:
    def __init__(self, total_iters, window_size=100):
        self.total_iters = total_iters
        # 使用滑动窗口监督 100 iter 的平均耗时
        self.time_window = deque(maxlen=window_size)
        self.last_time = time.time()

    def update(self):
        """ 必须在每次迭代后调用 """
        current_time = time.time()
        self.time_window.append(current_time - self.last_time)
        self.last_time = current_time

    def get_remaining(self, current_iter):
        if current_iter <= 0 or not self.time_window:
            return "0h0m"

        # 使用滑动窗口计算平均耗时
        avg_time = sum(self.time_window) / len(self.time_window)
        remaining = (self.total_iters - current_iter) * avg_time
        return self.format_time(remaining)

    @staticmethod
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = seconds % 3600 // 60
        return f"{hours}h{int(minutes)}m"

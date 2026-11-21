# 过滤设备曲线上的微小假峰：只报告有足够突出程度、且间距足够的周期冲击。

import numpy as np  # 按时间保存振动采样值。
from scipy.signal import find_peaks  # 找满足约束的局部峰位置。

trace = np.array([0, 0.2, 0, 1.4, 0.1, 0.7, 0, 0.2, 0, 1.6, 0.1, 0.3, 0, 1.3, 0])  # 有小毛刺和三次较明显的冲击。
all_peaks, _ = find_peaks(trace)  # 先找所有局部最大值作对比。
strong, properties = find_peaks(trace, prominence=0.9, distance=4)  # 同时限制突出程度和采样点间距。
print('全部局部峰', all_peaks.tolist())  # 原始峰含一些轻微波动。
print('保留的冲击位置', strong.tolist())  # 这还是采样点下标，不是秒。
print('保留峰的突出程度', properties['prominences'].round(2).tolist())  # 检查与筛选条件对应的峰属性。

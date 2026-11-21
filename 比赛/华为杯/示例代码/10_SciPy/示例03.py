# 用 MAD 找异常处理耗时：极大故障点会拉高均值与标准差；这里以中位数和 MAD 为口径报告异常。

import numpy as np  # 存放一批请求耗时并创建布尔掩码。
from scipy.stats import median_abs_deviation  # 计算稳健散布量。

elapsed = np.array([20, 21, 19, 22, 20, 23, 20, 21, 33, 170])  # 一般耗时与两个异常混合。
center = np.median(elapsed)  # 计算典型耗时，减少极端大值的影响。
mad = median_abs_deviation(elapsed)  # 默认用未缩放的绝对偏差中位数。
abnormal = elapsed > center + 5 * mad  # 本例业务规则，只标记异常长耗时。
print('中位数、MAD', float(center), float(mad))  # 这两个统计量由采样记录直接计算。
print('长耗时告警', elapsed[abnormal].tolist())  # 仅打印超过口径的原记录。

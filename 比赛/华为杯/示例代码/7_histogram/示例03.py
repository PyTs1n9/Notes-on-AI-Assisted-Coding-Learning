# 不等宽直方图的密度和箱内比例：区间宽度不同，密度值与「这一箱占几成」不是同一个数。

import numpy as np  # 计算分箱分布与各箱宽度。

# 服务器等待时间。
delay = np.array([0.1, 0.6, 0.7, 1.4, 2.3, 3.0, 3.1, 5.8])

# 数据范围内，箱宽依次变大。
bins = [0, 1, 3, 6]

# 默认得到箱内计数。
frequency, edges = np.histogram(delay, bins=bins)
print(frequency,edges)
# 设为 True 时返回密度。
density, _ = np.histogram(delay, bins=bins, density=True)
print(density,_)
# 这是每箱所包含的样本比例。
bin_fraction = frequency / frequency.sum()

# 密度乘本箱宽度才得到这一箱的比例。
area = density * np.diff(edges)

# 不按箱宽加权截图或错误比较。
print('箱内比例', bin_fraction.round(3).tolist())

# 与箱内比例不同。
print('各箱密度', density.round(3).tolist())

# 检查面积归一化。
print('密度乘箱宽后的合计', round(float(area.sum()), 6))

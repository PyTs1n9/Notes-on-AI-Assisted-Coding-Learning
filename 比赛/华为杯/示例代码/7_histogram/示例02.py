# 按质检区间统计厚度：使用不等宽业务区间，并识别边界内外的数据如何被分配。

import numpy as np  # 按设定边界做频数汇总。

thickness = np.array([4, 7, 8, 9, 10, 11, 15, 17, 21, 22])  # 样本包含多个端点与区间外值。
boundaries = [5, 8, 11, 16, 21]  # 预警和合格区间不是等宽的。
frequency, edges = np.histogram(thickness, bins=boundaries)  # 明确指定每一个边界。
outside = thickness.size - int(frequency.sum())  # 总数扣掉落箱数，得到区间外数量。
print('区间边界、计数', edges.tolist(), frequency.tolist())  # 最后一箱包含右边界 21。
print('未归入任何箱', outside)  # 不把范围外当作最边上的箱。

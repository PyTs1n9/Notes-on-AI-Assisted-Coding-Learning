# RGB 三通道各自计数：对小型彩色图逐通道取数，验证三个频数合计各是一个通道的像素数。

import numpy as np  # 保存像素并对强度分箱。

rgb = np.array([[[5, 55, 190], [80, 90, 210], [255, 13, 110]], [[155, 40, 18], [83, 10, 250], [0, 17, 245]]], dtype=np.uint8)  # 高×宽×红绿蓝。
totals = []  # 记录每个通道的总计数。
for channel, name in enumerate(['R', 'G', 'B']):  # 通道顺序已知为红绿蓝。
    values = rgb[:, :, channel]  # 取出该颜色的所有像素。
    counts, edges = np.histogram(values, bins=4, range=(0, 256))  # 8 位像素用统一的边界。
    totals.append(int(counts.sum()))  # 汇总，以便发现混通道或漏像素。
    print('通道、分箱计数', name, counts.tolist())  # 看颜色分布，不是同时表达空间位置。
print('箱边界', edges.tolist())  # 共有比箱数多 1 个边界。

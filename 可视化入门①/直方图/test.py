"""
直方图 · 练习
题目：记录了 60 个人的身高（cm），画直方图看分布。

要求：
    1. 用 plt.hist 画直方图，把数据分成 8 段（bins=8）
    2. 标题写“身高分布”，x 轴写“身高（cm）”，y 轴写“人数”
    （保存和显示的代码已经写好了，不用动）

写完后试着把 bins 改成 3 和 20 各运行一次，感受分段多少对图形的影响。
答案在文件最后。
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 数据（已给出，不用改）
heights = np.random.normal(170, 8, 60)  # 平均 170cm，共 60 人

# TODO 1：画直方图
# 写法参考：plt.hist(heights, bins=8)
plt.hist(heights, bins=20)

# TODO 2：加标题和坐标轴
plt.title('身高分布')
plt.xlabel('身高（cm）')
plt.ylabel('人数')

plt.savefig("直方图练习.png")
plt.show()

# ==================== 参考答案 ====================
# plt.hist(heights, bins=8)
# plt.title("身高分布")
# plt.xlabel("身高（cm）")
# plt.ylabel("人数")

"""
箱线图 · 练习
题目：两家工厂各抽检 40 个零件的长度（mm），比较谁的尺寸更稳定。

要求：
    1. 用 plt.boxplot 画出两组数据的箱线图（两组数据要放进一个列表里）
    2. 标题写“两家工厂零件长度对比”，y 轴写“长度（mm）”
    3. 用 plt.xticks([1, 2], ["工厂A", "工厂B"]) 把横轴改成厂名
    （保存和显示的代码已经写好了，不用动）

想一想：箱子越扁（高度越小）说明什么？异常点多说明什么？
答案在文件最后。
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 数据（已给出，不用改）
np.random.seed(1)
factory_a = np.random.normal(50.0, 0.6, 40)  # 波动小
factory_b = np.random.normal(50.2, 1.5, 40)  # 波动大
data = [factory_a, factory_b]

# TODO 1：画箱线图
# 写法参考：plt.boxplot(data)
plt.boxplot(data)

# TODO 2：加标题、y 轴标签
plt.title('两家工厂零件长度对比')
plt.ylabel('长度（mm）')

# TODO 3：把横轴刻度改成厂名
plt.xticks([1, 2], ["工厂A", "工厂B"])

plt.savefig("箱线图练习.png")
plt.show()

# ==================== 参考答案 ====================
# plt.boxplot(data)
# plt.title("两家工厂零件长度对比")
# plt.ylabel("长度（mm）")
# plt.xticks([1, 2], ["工厂A", "工厂B"])

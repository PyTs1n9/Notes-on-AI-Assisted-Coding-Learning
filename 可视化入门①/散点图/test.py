"""
散点图 · 练习
题目：某门店记录了 8 天的广告投入与营业额，看两者有没有关系。

要求：
    1. 用 plt.scatter 画出 广告投入 → 营业额 的散点图
    2. 标题写“广告投入与营业额”，x 轴写“广告投入（千元）”，y 轴写“营业额（千元）”
    （保存和显示的代码已经写好了，不用动）

写完后看一眼图：点是不是大致沿着一条斜线分布？
答案在文件最后。
"""

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 数据（已给出，不用改）
ad = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
revenue = [12, 15, 14, 19, 22, 21, 26, 28]

# TODO 1：画散点图
# 写法参考：plt.scatter(ad, revenue)
plt.scatter(ad, revenue, alpha=0.5)

# TODO 2：加标题和坐标轴
plt.title('广告投入与营业额')
plt.xlabel('广告投入')
plt.ylabel('营业额')
plt.savefig("散点图练习.png")
plt.show()

# ==================== 参考答案 ====================
# plt.scatter(ad, revenue)
# plt.title("广告投入与营业额")
# plt.xlabel("广告投入（千元）")
# plt.ylabel("营业额（千元）")

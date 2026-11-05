"""
柱状图 · 练习
题目：某水果店一天卖出 4 种水果，画出销量柱状图。

要求：
    1. 用 plt.bar 画出 水果 → 销量 的柱状图
    2. 标题写“水果销量”，x 轴写“水果”，y 轴写“销量（斤）”
    （保存和显示的代码已经写好了，不用动）

写完后运行：python test.py，看是否生成 柱状图练习.png
答案在文件最后。
"""

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 数据（已给出，不用改）
fruits = ["苹果", "香蕉", "橙子", "葡萄"]
amount = [35, 20, 28, 12]

# TODO 1：画柱状图
# 写法参考：plt.bar(fruits, amount)
plt.bar(fruits, amount)

# TODO 2：加标题和坐标轴
plt.title("水果销量")
plt.xlabel("水果")
plt.ylabel("销量（斤）")

plt.savefig("柱状图练习.png")
plt.show()


# ==================== 参考答案 ====================
# plt.bar(fruits, amount)
# plt.title("水果销量")
# plt.xlabel("水果")
# plt.ylabel("销量（斤）")

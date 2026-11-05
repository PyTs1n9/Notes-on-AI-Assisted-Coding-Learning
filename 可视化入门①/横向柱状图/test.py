"""
横向柱状图 · 练习
题目：统计 5 个部门的员工人数，用横向柱状图展示。

要求：
    1. 用 plt.barh 画出 部门 → 人数 的横向柱状图
    2. 标题写“各部门员工人数”，x 轴写“人数”，y 轴写“部门”
    （保存和显示的代码已经写好了，不用动）

小提醒：横向图的数值在 x 轴，类别在 y 轴，别写反。
答案在文件最后。
"""

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 数据（已给出，不用改）
departments = ["技术部", "市场部", "人事部", "财务部", "客服部"]
people = [42, 18, 9, 7, 25]

# TODO 1：画横向柱状图
# 写法参考：plt.barh(departments, people)
plt.barh(departments, people)

# TODO 2：加标题和坐标轴
plt.title('各部门员工人数')
plt.xlabel('部门')
plt.ylabel('人数')

plt.savefig("横向柱状图练习.png")
plt.show()

# ==================== 参考答案 ====================
# plt.barh(departments, people)
# plt.title("各部门员工人数")
# plt.xlabel("人数")
# plt.ylabel("部门")

"""
饼图 · 练习
题目：一个班级 20 人的出行方式统计，画饼图看各方式占比。

要求：
    1. 用 plt.pie 画出 出行方式 → 人数 的饼图，要显示标签和百分比（保留 1 位小数）
       写法提示：plt.pie(counts, labels=ways, autopct="%1.1f%%")
    2. 标题写“出行方式占比”
    （保存和显示的代码已经写好了，不用动）

想一想：饼图里能不能看出“坐地铁的有多少人”？看不出，它只显示比例。
答案在文件最后。
"""

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 数据（已给出，不用改）
ways = ["地铁", "公交", "骑车", "步行"]
counts = [9, 5, 4, 2]

# TODO 1：画饼图（记得显示标签和百分比）
plt.pie(counts,labels=ways,autopct='%1.1f%%')

# TODO 2：加标题
plt.title('出行方式占比')

plt.savefig("饼图练习.png")
plt.show()


# ==================== 参考答案 ====================
# plt.pie(counts, labels=ways, autopct="%1.1f%%")
# plt.title("出行方式占比")

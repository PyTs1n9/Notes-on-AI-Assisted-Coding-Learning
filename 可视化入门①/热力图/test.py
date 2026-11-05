"""
热力图 · 练习
题目：3 门课程 × 4 个学期的平均分，用热力图看哪门课、哪个学期分数高。

要求：
    1. 用 plt.imshow 画出二维数据
    2. 加颜色条（plt.colorbar()）
    3. 把横轴刻度改成学期名，纵轴刻度改成课程名
       写法提示：plt.xticks(range(4), ["大一上", "大一下", "大二上", "大二下"])
                 plt.yticks(range(3), ["数学", "英语", "编程"])
    4. 标题写“各学期课程平均分”
    （保存和显示的代码已经写好了，不用动）

提醒：数据是 3 行 4 列，所以 y 轴 3 个标签，x 轴 4 个标签，别弄反。
答案在文件最后。
"""

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 数据（3 行 = 3 门课，4 列 = 4 个学期）
data = [
	[78, 82, 85, 88],  # 数学
	[75, 77, 80, 83],  # 英语
	[70, 74, 81, 86],  # 编程
]

# TODO 1：画热力图
plt.imshow(data)

# TODO 2：加颜色条
plt.colorbar()

# TODO 3：设置横轴、纵轴刻度名字
plt.xticks(range(4), ['学期1', '学期2', '学期3', '学期4'])
plt.yticks(range(3), ['数学', '英语', '编程'])
# TODO 4：加标题

plt.savefig("热力图练习.png")
plt.show()

# ==================== 参考答案 ====================
# plt.imshow(data)
# plt.colorbar()
# plt.xticks(range(4), ["大一上", "大一下", "大二上", "大二下"])
# plt.yticks(range(3), ["数学", "英语", "编程"])
# plt.title("各学期课程平均分")

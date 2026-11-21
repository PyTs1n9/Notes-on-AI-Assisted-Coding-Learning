# 用箱线图检查同中位数不同波动：两条线路的典型送达时间可以一样，但其中一条明显更不稳定。

import os  # 确定图文件输出位置。
import matplotlib.pyplot as plt  # 保存绘图结果。
import pandas as pd  # 整理线路与送达时间。
import seaborn as sns  # 可按线路拆开画箱线图。

times = pd.DataFrame({'line': ['Stable'] * 7 + ['Variable'] * 7, 'minutes': [18, 19, 19, 20, 21, 21, 22, 7, 10, 15, 20, 26, 34, 51]})  # 各条线路都有一批记录。
folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')  # 让文件与示例放在一起。
os.makedirs(folder, exist_ok=True)  # 创建目录，不改原始数据。
fig, ax = plt.subplots(figsize=(5, 4))  # 创建一块坐标区。
sns.boxplot(data=times, x='line', y='minutes', ax=ax)  # 箱体跨越中间 50% 的数据。
ax.set_title('Delivery time by line')  # 这是耗时分布而非人数图。
fig.tight_layout()  # 自动调整边距。
fig.savefig(os.path.join(folder, 'delivery_spread.png'), dpi=120)  # 查看中位数线的位置与箱体在耗时轴上的高度。
plt.close(fig)  # 保存后关闭画布。
for line in ['Stable', 'Variable']:  # 逐线路提取与箱体含义相同的统计量。
    minutes = times.loc[times['line'] == line, 'minutes']  # 不把其他线路混进来。
    iqr = minutes.quantile(0.75) - minutes.quantile(0.25)  # IQR 是上四分位减下四分位。
    print('线路、中位数、IQR', line, minutes.median(), iqr)  # 分别检查典型耗时和波动。

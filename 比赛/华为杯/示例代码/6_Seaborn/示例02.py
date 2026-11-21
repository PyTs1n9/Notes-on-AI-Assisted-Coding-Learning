# 按行归一化的差错热图：组规模不同，比较每组类别占比而非原人数。

import os  # 保存图片。
import matplotlib.pyplot as plt  # 提供布局和保存。
import pandas as pd  # 记录不同行列的含义。
import seaborn as sns  # 对二维表画带标注的热图。

counts = pd.DataFrame({'Pass': [90, 18], 'Retry': [10, 12]}, index=['Line A', 'Line B'])  # 两条产线总件数不同。
rates = counts.div(counts.sum(axis=1), axis=0)  # 每行除以自己的总件数。
folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')  # 图保存在示例目录内。
os.makedirs(folder, exist_ok=True)  # 允许重复运行。
fig, ax = plt.subplots(figsize=(5, 4))  # 创建热图坐标区。
sns.heatmap(rates, annot=True, fmt='.0%', cmap='Blues', vmin=0, vmax=1, ax=ax)  # 百分数标注，色标固定 0 到 1。
ax.set_title('Outcome share within each line')  # 强调每一行是独立分母。
fig.tight_layout()  # 防止图例贴边。
fig.savefig(os.path.join(folder, 'line_rates.png'), dpi=120)  # 复核各行比例。
plt.close(fig)  # 关闭绘图窗口对象。
print('各线重试占比', rates['Retry'].round(3).to_dict())  # 输出图中关心的一列。

# 簇编号着色散点图：把同一批二维坐标按簇着色，避免误把坐标本身用于着色。

import os  # 保存到脚本旁的 outputs 目录。
import matplotlib.pyplot as plt  # 创建画布、保存并关闭。
import pandas as pd  # 组织样本坐标和组号。
import seaborn as sns  # 按数据列画散点。

points = pd.DataFrame({'pc1': [-2.1, -1.8, -1.7, 1.2, 1.5, 1.8], 'pc2': [0.5, 0.9, 0.3, -0.7, -0.3, 0.1], 'cluster': ['A', 'A', 'A', 'B', 'B', 'B']})  # 已有的二维投影结果。
folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')  # 不依赖当前工作目录。
os.makedirs(folder, exist_ok=True)  # 首次运行建目录。
fig, ax = plt.subplots(figsize=(6, 4))  # 一张画布一个坐标区。
sns.scatterplot(data=points, x='pc1', y='pc2', hue='cluster', palette='Set2', s=75, ax=ax)  # 簇名决定颜色。
ax.set_title('Projection by cluster')  # 图题说明点的来源。
fig.tight_layout()  # 给坐标标签留位置。
fig.savefig(os.path.join(folder, 'clusters.png'), dpi=120)  # 供复查。
plt.close(fig)  # 释放画布资源。
print('每组样本数', points['cluster'].value_counts().to_dict())  # 图和计数来自同一批样本。

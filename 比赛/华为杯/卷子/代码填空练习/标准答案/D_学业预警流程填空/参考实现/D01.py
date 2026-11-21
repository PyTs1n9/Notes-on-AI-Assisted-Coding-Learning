# D01：参考实现。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "D_学业预警流程填空", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "D01")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import matplotlib.pyplot as plt
import pandas as pd

raw = pd.read_csv(os.path.join(DATA, 'campus_raw.csv'))
print('原表行列', raw.shape)
raw.info()
# 缺失统计面向刚读到的全表，与后面的有效成绩分母不同。
missing = raw.isna().sum()
missing.rename_axis('field').reset_index(name='missing').to_csv(os.path.join(OUT, 'missing.csv'), index=False)
# 区间条件包含 0 和 20，自然排除缺失；不能对目标值随便插补。
legal = raw.loc[raw['G3'].between(0, 20), 'G3']
# 把四个统计量一次应用在合法成绩序列上。
summaries = legal.agg(['max', 'min', 'mean', 'median'])
summaries.rename_axis('stat').reset_index(name='value').to_csv(os.path.join(OUT, 'final_stats.csv'), index=False)
labels = ['0-<5', '5-<10', '10-<15', '15-20']
# 分箱左闭右开；由于此前先排除超过 20 的值，最后箱用右边界 21 将 20 纳入。
bands = pd.cut(legal, bins=[0, 5, 10, 15, 21], right=False, labels=labels)
counts = bands.value_counts(sort=False)
counts.rename_axis('band').reset_index(name='count').to_csv(os.path.join(OUT, 'score_bands.csv'), index=False)
ax = counts.plot(kind='bar')
ax.set(xlabel='Score band', ylabel='Students')
ax.figure.tight_layout()
ax.figure.savefig(os.path.join(OUT, 'score_bands.png'), dpi=120)
plt.close(ax.figure)
print('有效成绩最小值和最大值', legal.min(), legal.max(), '各箱计数', counts.tolist())

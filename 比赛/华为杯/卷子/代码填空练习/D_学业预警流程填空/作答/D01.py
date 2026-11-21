# D01：只替换带题号的空位。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "D01")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import matplotlib.pyplot as plt
import pandas as pd

raw = pd.read_csv(os.path.join(DATA, 'campus_raw.csv'))
print('原表行列', raw.shape)
raw.info()
missing = __D01_01__
missing.rename_axis('field').reset_index(name='missing').to_csv(os.path.join(OUT, 'missing.csv'), index=False)
legal = raw.loc[__D01_02__, 'G3']
summaries = legal.agg(__D01_03__)
summaries.rename_axis('stat').reset_index(name='value').to_csv(os.path.join(OUT, 'final_stats.csv'), index=False)
labels = ['0-<5', '5-<10', '10-<15', '15-20']
bands = pd.cut(legal, __D01_04__, labels=labels)
counts = bands.value_counts(sort=False)
counts.rename_axis('band').reset_index(name='count').to_csv(os.path.join(OUT, 'score_bands.csv'), index=False)
ax = counts.plot(kind='bar')
ax.set(xlabel='Score band', ylabel='Students')
ax.figure.tight_layout()
ax.figure.savefig(os.path.join(OUT, 'score_bands.png'), dpi=120)
plt.close(ax.figure)
print('有效成绩最小值和最大值', legal.min(), legal.max(), '各箱计数', counts.tolist())

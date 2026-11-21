# A05：只替换带题号的空位。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "A05")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import matplotlib.pyplot as plt
import pandas as pd

frame = pd.read_csv(os.path.join(DATA, 'scores.csv'))
print('行列', frame.shape)
frame.__A05_01__()
columns = ['math', 'language', 'practice']
quality = pd.DataFrame({'column': frame.columns, 'dtype': frame.dtypes.astype(str).to_numpy(),
                        'non_null': frame.count().to_numpy(), 'missing': __A05_02__})
quality.to_csv(os.path.join(OUT, 'quality.csv'), index=False)
ax = quality.plot(x='column', y='missing', kind=__A05_03__, color='#FF6B6B', legend=False)
ax.set_ylabel('Missing values')
ax.figure.tight_layout()
ax.figure.savefig(os.path.join(OUT, 'missing.png'), dpi=120)
plt.close(ax.figure)
abnormal_parts = []
counts = []
for column in columns:
    values = frame[column]
    bad = __A05_04__
    hits = frame.loc[bad, ['id', column]].copy()
    hits['row_no'] = hits.index + 1
    hits['field'] = column
    hits = hits.rename(columns={column: 'value'})
    abnormal_parts.append(hits)
    counts.append(int(bad.sum()))
abnormal = pd.__A05_05__(abnormal_parts, ignore_index=True)
dup = frame.loc[frame.duplicated(__A05_06__)].copy()
dup['row_no'] = dup.index + 1
abnormal.to_csv(os.path.join(OUT, 'abnormal.csv'), index=False)
pd.DataFrame({'field': columns, 'count': counts}).to_csv(os.path.join(OUT, 'abnormal_counts.csv'), index=False)
dup.to_csv(os.path.join(OUT, 'duplicates.csv'), index=False)
print('越界数', counts, '重复涉及记录数', len(dup))

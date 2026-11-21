# A05：参考实现。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "A_参考案例全覆盖", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "A05")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import matplotlib.pyplot as plt
import pandas as pd

frame = pd.read_csv(os.path.join(DATA, 'scores.csv'))
print('行列', frame.shape)
# info 直接输出类型与非空数量，不要把它的返回值作为统计表。
frame.info()
columns = ['math', 'language', 'practice']
quality = pd.DataFrame({'column': frame.columns, 'dtype': frame.dtypes.astype(str).to_numpy(),
                        # isnull 得到逐单元布尔值，再计算各列缺失数。
                        'non_null': frame.count().to_numpy(), 'missing': frame.isnull().sum().to_numpy()})
quality.to_csv(os.path.join(OUT, 'quality.csv'), index=False)
# 一列一个柱子，满足离散字段对比。
ax = quality.plot(x='column', y='missing', kind='bar', color='#FF6B6B', legend=False)
ax.set_ylabel('Missing values')
ax.figure.tight_layout()
ax.figure.savefig(os.path.join(OUT, 'missing.png'), dpi=120)
plt.close(ax.figure)
abnormal_parts = []
counts = []
for column in columns:
    values = frame[column]
    # 只要落在任一越界方向即可，边界 0、100 合法。
    bad = (values < 0) | (values > 100)
    hits = frame.loc[bad, ['id', column]].copy()
    hits['row_no'] = hits.index + 1
    hits['field'] = column
    hits = hits.rename(columns={column: 'value'})
    abnormal_parts.append(hits)
    counts.append(int(bad.sum()))
# 按字段做成长表后纵向合并，保留同一记录的多个异常项。
abnormal = pd.concat(abnormal_parts, ignore_index=True)
# 默认只标后续副本；keep=False 才标出每组全部成员。
dup = frame.loc[frame.duplicated(keep=False)].copy()
dup['row_no'] = dup.index + 1
abnormal.to_csv(os.path.join(OUT, 'abnormal.csv'), index=False)
pd.DataFrame({'field': columns, 'count': counts}).to_csv(os.path.join(OUT, 'abnormal_counts.csv'), index=False)
dup.to_csv(os.path.join(OUT, 'duplicates.csv'), index=False)
print('越界数', counts, '重复涉及记录数', len(dup))

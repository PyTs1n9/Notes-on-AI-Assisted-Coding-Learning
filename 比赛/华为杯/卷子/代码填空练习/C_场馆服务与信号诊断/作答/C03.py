# C03：只替换带题号的空位。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "C03")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import matplotlib.pyplot as plt
import pandas as pd

raw = pd.read_csv(os.path.join(DATA, 'bookings.csv'), dtype={'day': str, 'minutes': str})
correct_shape = raw['day'].str.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', na=False)
days = pd.to_datetime(raw['day'].where(correct_shape), format='%Y-%m-%d', errors=__C03_01__)
minutes = pd.to_numeric(raw['minutes'], errors=__C03_02__)
masks = [('day', 'missing', raw['day'].isna()),
         ('day', 'bad_format', __C03_03__),
         ('minutes', 'missing', raw['minutes'].isna()),
         ('minutes', 'bad_format', __C03_04__),
         ('minutes', 'out_of_range', __C03_05__)]
issues = []
for field, reason, mask in masks:
    selected = raw.loc[mask, ['id', field]].copy()
    selected['field'] = field
    selected['reason'] = reason
    issues.append(selected.rename(columns={field: 'raw_value'}))
details = pd.concat(issues, ignore_index=True)
valid_mask = __C03_06__
valid = pd.DataFrame({'id': raw['id'], 'day': days, 'minutes': minutes}).loc[valid_mask].copy()
tally = details['reason'].value_counts().rename_axis('reason').reset_index(name='count')
ax = tally.plot(x='reason', y='count', kind='bar', legend=False)
ax.set_ylabel('Issues')
ax.figure.tight_layout()
ax.figure.savefig(os.path.join(OUT, 'booking_issues.png'), dpi=120)
plt.close(ax.figure)
details.to_csv(os.path.join(OUT, 'booking_issues.csv'), index=False)
valid.to_csv(os.path.join(OUT, 'valid.csv'), index=False)
tally.to_csv(os.path.join(OUT, 'issue_tally.csv'), index=False)
print('各类问题次数', tally.set_index('reason')['count'].to_dict())
print('通过记录', valid['id'].tolist())

# D02：只替换带题号的空位。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "D02")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import pandas as pd

raw = pd.read_csv(os.path.join(DATA, 'campus_raw.csv'))
nonempty = raw.dropna(how=__D02_01__)
frame = nonempty.loc[__D02_02__].copy()
valid_abs = frame['absences'].between(0, 30)
replacement = __D02_03__
abs_fixed = frame.loc[__D02_04__, ['id', 'absences']].copy()
frame.loc[~valid_abs, 'absences'] = replacement
frame['study_hours'] = frame['study_hours'].fillna(__D02_05__)
for col in ['school', 'support']:
    frame[col] = frame[col].fillna(__D02_06__)
frame['risk'] = __D02_07__
counts = frame['risk'].value_counts().sort_index()
frame.to_csv(os.path.join(OUT, 'clean_campus.csv'), index=False)
abs_fixed.to_csv(os.path.join(OUT, 'absence_fixed.csv'), index=False)
counts.rename_axis('risk').reset_index(name='count').to_csv(os.path.join(OUT, 'risk_counts.csv'), index=False)
print('删除行数', len(raw) - len(frame), '修复缺勤数', len(abs_fixed))
print('标签计数', counts.to_dict())

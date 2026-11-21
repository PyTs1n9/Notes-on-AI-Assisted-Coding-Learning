# D02：参考实现。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "D_学业预警流程填空", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "D02")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import pandas as pd

raw = pd.read_csv(os.path.join(DATA, 'campus_raw.csv'))
# 只删全空行；不能误用任意字段缺失就删行。
nonempty = raw.dropna(how='all')
# 标签要来自真实合法成绩，不能先插补 G3 再设标签。
frame = nonempty.loc[nonempty['G3'].between(0, 20)].copy()
valid_abs = frame['absences'].between(0, 30)
# 排除不合法缺勤及缺失：不能用被污染的全列算修复中位数。
replacement = frame.loc[valid_abs, 'absences'].median()
# 合法掩码取反会同时选择越界和缺失缺勤。
abs_fixed = frame.loc[~valid_abs, ['id', 'absences']].copy()
frame.loc[~valid_abs, 'absences'] = replacement
# 数值型学习时长使用当前保留人群的有效均值。
frame['study_hours'] = frame['study_hours'].fillna(frame['study_hours'].mean())
for col in ['school', 'support']:
    # mode 可能返回多个值，本题明确选首个给 fillna 一个标量。
    frame[col] = frame[col].fillna(frame[col].mode().iloc[0])
# 严格小于阈值才标 1，10 为 0，0 合法且为 1。
frame['risk'] = (frame['G3'] < 10).astype(int)
counts = frame['risk'].value_counts().sort_index()
frame.to_csv(os.path.join(OUT, 'clean_campus.csv'), index=False)
abs_fixed.to_csv(os.path.join(OUT, 'absence_fixed.csv'), index=False)
counts.rename_axis('risk').reset_index(name='count').to_csv(os.path.join(OUT, 'risk_counts.csv'), index=False)
print('删除行数', len(raw) - len(frame), '修复缺勤数', len(abs_fixed))
print('标签计数', counts.to_dict())

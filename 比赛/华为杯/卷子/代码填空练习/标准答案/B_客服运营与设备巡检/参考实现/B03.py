# B03：参考实现。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "B_客服运营与设备巡检", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "B03")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import pandas as pd

frame = pd.read_csv(os.path.join(DATA, 'meter_log.csv'))
missing = frame.isna().sum().rename_axis('field').reset_index(name='missing')
original_columns = frame.columns.tolist()
frame['row_no'] = frame.index + 1
# 标记全组；新增的行序号不可放进比较字段。
exact_mask = frame.duplicated(subset=original_columns, keep=False)
exact = frame.loc[exact_mask].copy()
# 先去掉真正的内容重复，才能区分下一步的同键不同内容。
unique = frame.drop_duplicates(subset=original_columns, keep='first')
key = ['device', 'slot']
# 按业务键而非全列判断，且列出同组各成员。
conflict = unique.loc[unique.duplicated(subset=key, keep=False)]
limits = {'load_pct': (0, 100), 'humidity_pct': (0, 100), 'temp_c': (-20, 60)}
records = []
for field, pair in limits.items():
    low, high = pair
    values = unique[field]
    # 合法端点不报错；数值缺失与任一不等式比较为 False。
    bad = (values < low) | (values > high)
    hits = unique.loc[bad, ['row_no', field]].copy()
    hits['field'] = field
    hits = hits.rename(columns={field: 'value'})
    records.append(hits)
# 纵向保留字段明细，避免同一行多字段越界时被合并丢失。
abnormal = pd.concat(records, ignore_index=True)
missing.to_csv(os.path.join(OUT, 'meter_missing.csv'), index=False)
exact.to_csv(os.path.join(OUT, 'exact_duplicate_rows.csv'), index=False)
conflict.to_csv(os.path.join(OUT, 'key_conflict_rows.csv'), index=False)
abnormal.to_csv(os.path.join(OUT, 'out_of_range.csv'), index=False)
print('完整重复涉及行', exact['row_no'].tolist())
print('键冲突涉及行', conflict['row_no'].tolist())
print('越界字段次数', abnormal['field'].value_counts().sort_index().to_dict())

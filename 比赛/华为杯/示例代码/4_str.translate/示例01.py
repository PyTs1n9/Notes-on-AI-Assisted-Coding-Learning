# 只删除编号中的分隔字符：保留原编号列，同时生成用于检索的紧凑版本。
# 示范一列编号一起清理。
import pandas as pd

# 包含缺失记录和多类分隔符。
original = pd.Series(['AB-01/02', 'XY-20', None, 'SN/8-9'])

# 第三个参数中的连字符和斜线直接删除。
drop = str.maketrans('', '', '-/')

# 对一列逐值处理，缺失仍为缺失。
compact = original.str.translate(drop)

# 原数据不受影响。
print('原编号', original.tolist())

# 单独查看有内容的清洗结果。
print('检索时使用的编号', compact.dropna().tolist())

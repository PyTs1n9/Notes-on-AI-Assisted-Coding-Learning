# 按字面搜索版本片段：查确切的版本字符串，别让小数点匹配任意字符。

import pandas as pd  # 按列查找字符串。

logs = pd.Series(['client v1.2 ready', 'client v1x2 ready', None, 'v2.1 ok'])  # 有近似但错误的匹配项。
exact = logs.str.contains('v1.2', regex=False, na=False)  # 点号按字面匹配。
pattern = logs.str.contains('v1.2', na=False)  # 对照：默认正则里点号有别的含义。
print('字面命中', logs[exact].tolist())  # 业务所需的筛选结果。
print('默认正则命中', logs[pattern].tolist())  # 观察误匹配。

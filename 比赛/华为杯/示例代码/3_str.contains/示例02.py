# 筛选有格式的报警编码：查形如 ERR-003 的独立片段，不匹配 ERR-3 或更长数字。

import pandas as pd  # 支持矢量化文本筛选。

frame = pd.DataFrame(
	{'log':
		 ['stop err-003 now', 'ERR-1234', 'ERR-7', 'alert ERR-902.', None]
	 }
)  # 小批日志。

mask = frame['log'].str.contains(r'\bERR-[0-9]{3}\b', case=False, regex=True, na=False)  # 三位数字，且两端是词边界。
print(mask)
hits = frame.loc[mask, 'log']  # 只取命中的原日志。
print(hits)
print('符合编码格式', hits.tolist())  # 输出包含完整编码的记录。

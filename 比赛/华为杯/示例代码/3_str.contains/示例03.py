# 包含故障但排除演练工单：把两条条件组合：文字含故障或停机，同时不属于演练或测试。

import pandas as pd  # 生成数据表并组合按行筛选条件。

# 演练和真实异常混在一起。
frame = pd.DataFrame(
	{
		'id': [11, 12, 13, 14, 15],
		'text': ['机房停机', '测试时停机', '已有演练处理故障', '屏幕故障', None]
	}
)

# 正则中的竖线表示任意一个词。
include = frame['text'].str.contains('故障|停机', na=False)
print(include)
# 独立识别应排除内容。
exclude = frame['text'].str.contains('演练|测试', na=False)
print(exclude)

# 同时满足「命中」和「未排除」。
selected = frame.loc[include & ~exclude, 'id']

# 看输出的是记录编号，不是默认行下标。
print('真实异常编号', selected.tolist())

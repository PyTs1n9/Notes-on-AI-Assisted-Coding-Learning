# 比较两个预先给定的预警阈值：同一个模型，不同阈值会改变报警的数量；这里只观察差异，不用测试数据选阈值。

import numpy as np  # 用于比较预测和标签的布尔条件。
from sklearn.linear_model import LogisticRegression  # 学习高风险概率。

# 训练特征：累计异常次数。
historical = [
	[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]
]
# 真实记录允许少量重复或反常模式。
labels = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
# 用于示例的独立观察集。
upcoming = [[2], [3], [4], [5], [6], [7]]
# 正类是实际需要关注的记录。
truth = np.array([0, 0, 0, 1, 1, 1])
# 使用同一个已训练模型。
model = LogisticRegression(max_iter=300)
# 只用历史样本拟合。
model.fit(historical, labels)
# 查找高风险在概率矩阵中的列位置。
column = list(model.classes_).index(1)
# 先取每条记录的同一类概率。
risk = model.predict_proba(upcoming)[:, column]

# 两个演示阈值事先指定。
for cutoff in [0.35, 0.65]:
	flagged = risk >= cutoff  # 达到阈值才报告。
	missed = int((~flagged & (truth == 1)).sum())  # 没报但其实是正类的数量。
	false_alarm = int((flagged & (truth == 0)).sum())  # 报了但其实是负类的数量。
	print('阈值、漏报、误报', cutoff, missed, false_alarm)  # 把代价放在同一口径上观察。

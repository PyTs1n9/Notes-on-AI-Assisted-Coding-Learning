# 多分类概率按类名取列：将三类请求分给对应窗口，同时正确读出「缴费」类的概率，而不是固定列下标。

import numpy as np  # 用于找到逐行最大的概率列。
from sklearn.linear_model import LogisticRegression  # 同时预测三个类别。

train = [[3, 0], [4, 1], [5, 0], [0, 4], [1, 5], [0, 6], [4, 4], [5, 5], [6, 4]]  # 两项已整理的计数特征。
target = ['报修', '报修', '报修', '缴费', '缴费', '缴费', '咨询', '咨询', '咨询']  # 三类各有训练记录。
model = LogisticRegression(max_iter=400)  # 创建三分类任务的模型。
model.fit(train, target)  # 学习特征和类别的对应关系。
new = [[0, 5], [5, 5], [5, 1]]  # 三条新的请求。
probs = model.predict_proba(new)  # 每条请求得到三个概率。
labels = model.classes_[np.argmax(probs, axis=1)]  # 各条请求取概率最大的类名。
pay_col = list(model.classes_).index('缴费')  # 查询指定类对应的概率列。
print('概率列顺序', model.classes_.tolist())  # 列顺序不一定等于训练数据中出现顺序。
print('自动分配', labels.tolist())  # 用类名展示分流结果。
print('各条缴费概率', probs[:, pay_col].round(4).tolist())  # 缴费概率可能不是最后一列。

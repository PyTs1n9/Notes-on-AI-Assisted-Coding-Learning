# 订单是否需人工复核：在训练集上拟合预处理和分类器，在未参与训练的订单上看类别和准确率。

import pandas as pd  # 用列名记录建模特征。
from sklearn.linear_model import LogisticRegression  # 二分类模型。
from sklearn.metrics import accuracy_score  # 比较真实和预测类别。
from sklearn.pipeline import make_pipeline  # 串起两个训练步骤。
from sklearn.preprocessing import StandardScaler  # 学习两列各自的尺度。

# 已有历史特征。
X_train = pd.DataFrame({
	'items': [1, 2, 2, 3, 5, 6, 7, 9],
	'amount': [20, 30, 49, 52, 90, 115, 140, 180]
})
# 独立测试订单。
X_test = pd.DataFrame({
	'items': [3, 8, 2, 6],
	'amount': [45, 155, 25, 135]
})
# 1 表示需要人工复核。
y_train = [0, 0, 0, 0, 1, 1, 1, 1]
# 测试答案只用于评估。
y_test = [0, 1, 0, 1]

# 先按训练样本标准化，再拟合分类器。
model = make_pipeline(
	StandardScaler(),
	LogisticRegression(max_iter=300)
)

# 测试样本不参与学习均值、尺度和分类参数。
model.fit(X_train, y_train)

# 管道自动复用训练尺度。
predicted = model.predict(X_test)

# 一项对应一笔测试订单。
print('复核类别', predicted.tolist())

# 预测正确占比。
print('测试准确率', round(accuracy_score(y_test, predicted), 4))

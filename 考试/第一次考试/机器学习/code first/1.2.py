import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

X, y = load_wine(return_X_y=True)

# 使训练集和测试集中三个品种的比例与原数据保持一致
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42, stratify=y
)

print('训练集形状:', X_train.shape, '测试集形状:', X_test.shape)
print('原数据类别占比:', np.round(np.bincount(y) / len(y), 3))
print('训练集类别占比:', np.round(np.bincount(y_train) / len(y_train), 3))

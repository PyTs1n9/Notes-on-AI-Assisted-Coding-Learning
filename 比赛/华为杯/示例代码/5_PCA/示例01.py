# 由解释方差比例决定保留几列：三个互相关联的传感特征是否能用更少的轴表达？

import numpy as np  # 用数组准备小型数值表。
from sklearn.decomposition import PCA  # 提取变化最多的坐标方向。

history = np.array([[1, 2, 1.1], [2, 4, 2.2], [3, 6, 2.9], [4, 8, 4.1], [5, 10, 4.9]])  # 三列已在相近尺度，第二列与第一列相关。
pca = PCA(n_components=0.95, svd_solver='full')  # 选择能累计保留至少约 95% 方差的最少轴。
reduced = pca.fit_transform(history)  # 先学习坐标系，再投影这批数据。
print('原始形状、降维形状', history.shape, reduced.shape)  # 观察保留多少列；样本数不变。
print('保留的方差比例', round(float(pca.explained_variance_ratio_.sum()), 4))  # 这不是模型准确率。

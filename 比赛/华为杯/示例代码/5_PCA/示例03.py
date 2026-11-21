# 同一数据先不缩放再缩放：当一列的数值单位特别大时，比较它会不会独占第一主成分。

import numpy as np  # 对数组计算方向系数和方差。
from sklearn.decomposition import PCA  # 比较一维投影方向。
from sklearn.preprocessing import StandardScaler  # 使不同单位的特征可比较。

raw = np.array([[850, 0.12], [900, 0.21], [1010, 0.18], [1120, 0.32], [1180, 0.30], [1300, 0.41]])  # 明显不同的量纲。
pca_raw = PCA(n_components=1).fit(raw)  # 不做缩放，直接观察主轴。
scaled = StandardScaler().fit_transform(raw)  # 每列在各自尺度上标准化。
pca_scaled = PCA(n_components=1).fit(scaled)  # 再看标准化特征的主轴。
print('原单位主轴的系数绝对值', np.abs(pca_raw.components_[0]).round(4).tolist())  # 忽略轴整体可能翻转的正负号。
print('标准化后主轴的系数绝对值', np.abs(pca_scaled.components_[0]).round(4).tolist())  # 判断两列谁在支配方向。

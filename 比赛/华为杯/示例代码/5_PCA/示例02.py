# 新记录复用历史坐标轴：将今日记录加入历史投影图时，不在今日样本上重新计算尺度和坐标。

import numpy as np  # 用二维数组提供历史和今日特征。
from sklearn.decomposition import PCA  # 提供两维投影。
from sklearn.preprocessing import StandardScaler  # 统一访问次数、分钟数等尺度。

history = np.array([[2, 52, 5], [4, 48, 8], [9, 21, 15], [10, 17, 19], [15, 12, 33]])  # 历史的三列行为数据。
today = np.array([[4, 48, 8], [11, 15, 22]])  # 第一行刻意与一条历史记录完全相同。
scaler = StandardScaler()  # 学习历史各列的均值与标准差。
historical_scaled = scaler.fit_transform(history)  # 只拟合历史特征。
today_scaled = scaler.transform(today)  # 今日仍使用历史尺度。
pca = PCA(n_components=2)  # 保留用于可视化的两维。
old_points = pca.fit_transform(historical_scaled)  # 根据历史位置建立坐标系。
new_points = pca.transform(today_scaled)  # 将今日特征投到同一坐标系。
print('历史与今日的坐标形状', old_points.shape, new_points.shape)  # 两批行数可以不同，列数相同。
print('相同记录投影重合', np.allclose(old_points[1], new_points[0]))  # 用同一点测试坐标能否直接比较。

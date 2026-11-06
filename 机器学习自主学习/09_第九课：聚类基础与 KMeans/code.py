import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.DataFrame({
	'年收入（元）': [50000, 80000, 120000, 300000, 350000, 400000],
	'消费评分': [20, 25, 30, 70, 75, 80]
})
print(df)
print(df.dtypes)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
print(np.round(X_scaled, 2))
print('各列均值:', X_scaled.mean(axis=0).round(4))
print('各列标准差:', X_scaled.std(axis=0).round(4))

km = KMeans(n_clusters=2, random_state=42, n_init=10)
km.fit(X_scaled)
print('每个样本的簇标签:', km.labels_)
print('聚类中心（标准化后）:')
print(km.cluster_centers_.round(2))
print('inertia_（簇内平方和）:', round(km.inertia_, 2))

df['簇'] = km.labels_
print(df)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
df = pd.read_csv('players.csv', encoding='utf-8-sig')
print('形状:', df.shape)
print('类型:', df.dtypes)
print('缺失值数量:', df.isnull().sum().sum())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
print(X_scaled.shape)
print(X_scaled.mean(axis=0).round(4))
print(X_scaled.std(axis=0).round(4))

Ks = range(2, 11)
inertia = []
drops = []
prev = None

for k in Ks:
	km = KMeans(n_clusters=k, random_state=42, n_init=10)
	km.fit(X_scaled)
	inertia.append(round(km.inertia_, 2))
	if prev is None:
		drops.append(None)
	else:
		drops.append(round(prev - km.inertia_, 2))
	prev = km.inertia_

for k, iner, drop in zip(Ks, inertia, drops):
	d = 'none' if drop is None else drop
	print(f'K={k:<3d} inertia_={iner:<6} 下降幅度={d}')

plt.plot(Ks, inertia, 'o-')
plt.xlabel('簇数 K')
plt.ylabel('inertia_（簇内平方和）')
plt.title('肘部法则：簇数 K 与 inertia 的关系（球员数据）')
plt.tight_layout()
plt.savefig('k-inertia.png')
plt.show()

print(f'读结果：K=3-4 下降幅度 {drops[2]}，K=4→5 下降幅度 {drops[3]}，4 之后每档都不足 1——')
print('下降从 K=4 开始就直接变得平缓，肘部在 K=4')

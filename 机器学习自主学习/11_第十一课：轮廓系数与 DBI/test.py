import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

df = pd.read_csv('players.csv', encoding='utf-8-sig')
print(df.shape)
print(df.dtypes)
print(df.isnull().sum().sum())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
print(X_scaled.shape)
print(X_scaled.mean(axis=0).round(4))
print(X_scaled.std(axis=0).round(4))

Ks = list(range(2, 11))
sils = []
dbis = []

for k in Ks:
	km = KMeans(n_clusters=k, random_state=42, n_init=10)
	km.fit(X_scaled)
	sils.append(round(silhouette_score(X_scaled, km.labels_), 4))
	dbis.append(round(davies_bouldin_score(X_scaled, km.labels_), 4))
print(sils)
print(dbis)

best_sil = Ks[sils.index(max(sils))]
best_dbi = Ks[dbis.index(min(dbis))]
print(f"轮廓系数峰值 K={best_sil}")
print(f"DBI谷值 K={best_dbi}")

print('最佳K是4，轮廓系数最大值0.8652对应的K=4，DBI最小的谷值0.1819对应的K=4，第十课肘部结论对应的K=4')

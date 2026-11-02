import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
	davies_bouldin_score, silhouette_score
)

df = pd.read_csv('../data_outputs/day_clean.csv', encoding='utf-8-sig')
X = df[['temp', 'hum', 'windspeed', 'cnt']]
Ks = list(range(2, 9))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
sils = []
dbis = []

for k in Ks:
	km = KMeans(n_clusters=k, random_state=42, n_init=10)
	km.fit(X_scaled)

	inertia.append(km.inertia_)

	sil_score = silhouette_score(X_scaled, km.labels_)
	sils.append(sil_score)

	dbi_score = davies_bouldin_score(X_scaled, km.labels_)
	dbis.append(dbi_score)

kmeans_scores = pd.DataFrame({
	'K': Ks,
	'inertia': inertia,
	'sils': sils,
	'dbis': dbis
})
kmeans_scores.to_csv('../metrics/kmeans_scores.csv', encoding='utf-8-sig', index=False)

plt.plot(Ks, inertia, 'o--')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Inertia - K')
plt.savefig('../figures/elbow_curve.png', dpi=150)
plt.close()

plt.plot(Ks, sils, 'o--')
plt.xlabel('K')
plt.ylabel('silhouette_score')
plt.title('silhouette_score - K')
plt.savefig('../figures/silhouette_score.png', dpi=150)
plt.close()

plt.plot(Ks, dbis, 'o--')
plt.xlabel('K')
plt.ylabel('davies_bouldin_score')
plt.title('davies_bouldin_score - K')
plt.savefig('../figures/davies_bouldin_score.png', dpi=150)
plt.close()

km = KMeans(n_clusters=4, random_state=42, n_init=10)
km.fit(X_scaled)
labels = km.labels_
plt.scatter(df['temp'], df['cnt'], c=labels, cmap="tab20", alpha=0.5)
plt.xlabel('temp')
plt.ylabel('cnt')
plt.title('temp - cnt')
plt.savefig('../figures/kmeans_scatter.png', dpi=150)
plt.close()

# print('轮廓系数最大峰值对应K为4，DBI最小谷值对应K为4，肘部法则肉眼看图K为3或4，结合轮廓系数和DBI确定最终K=4')

df['cluster'] = km.labels_
daily_clusters = df[['dteday', 'cluster']]
daily_clusters.to_csv('../data_outputs/daily_clusters.csv', index=False, encoding='utf-8-sig')

cluster_summary = df.groupby('cluster')[['temp', 'hum', 'windspeed', 'cnt']].mean().round(6)
cluster_summary['count'] = df.groupby('cluster').size()
cluster_summary = cluster_summary.reset_index()
cluster_summary.to_csv('../data_outputs/cluster_summary.csv', index=False, encoding='utf-8-sig')


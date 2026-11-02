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

daily_clusters = pd.DataFrame({'dteday': df['dteday'], 'cluster': labels})
daily_clusters.to_csv('../data_outputs/daily_clusters.csv', index=False, encoding='utf-8-sig')
print('--- daily_clusters.csv ---')
print(daily_clusters.head().to_string(index=False), ' 行数 =', len(daily_clusters))

cluster_list = []
temp_list = []
hum_list = []
wind_list = []
cnt_list = []
count_list = []

for c in range(4):                                    # c = 0,1,2,3 逐个簇算
	mask = (labels == c)                              # 731 个 True/False，标出哪些天属于簇 c
	cluster_list.append(c)
	temp_list.append(round(df.loc[mask, 'temp'].mean(), 6))          # 该簇 temp 均值
	hum_list.append(round(df.loc[mask, 'hum'].mean(), 6))            # 该簇 hum 均值
	wind_list.append(round(df.loc[mask, 'windspeed'].mean(), 6))     # 该簇 windspeed 均值
	cnt_list.append(round(df.loc[mask, 'cnt'].mean(), 6))            # 该簇 cnt 均值
	count_list.append(int(mask.sum()))                # True 当 1 加起来 = 该簇天数

cluster_summary = pd.DataFrame({
	'cluster': cluster_list,
	'temp': temp_list,
	'hum': hum_list,
	'windspeed': wind_list,
	'cnt': cnt_list,
	'count': count_list
})
cluster_summary.to_csv('../data_outputs/cluster_summary.csv', index=False, encoding='utf-8-sig')
print('--- cluster_summary.csv ---')
print(cluster_summary.to_string(index=False))

# ---------- 六、A3-5 命名依据：先算基线，再逐簇对比 ----------
base_temp = df['temp'].mean()                         # 基线 = 全表均值（“平均水平”长什么样）
base_hum = df['hum'].mean()
base_wind = df['windspeed'].mean()
base_cnt = df['cnt'].mean()
print('--- 命名依据 ---')
print('全表均值: temp =', round(base_temp, 4), ' hum =', round(base_hum, 4),
      ' windspeed =', round(base_wind, 4), ' cnt =', round(base_cnt, 1))

for c in range(4):
	mask = (labels == c)
	n = int(mask.sum())
	print('--- 簇', c, '：共', n, '天，占', round(100 * n / len(df), 1), '% ---')
	for f in ['temp', 'hum', 'windspeed', 'cnt']:
		value = df.loc[mask, f].mean()                # 这个簇该特征的均值
		base = df[f].mean()                           # 全表该特征的均值
		pct = (value / base - 1) * 100                # 比基线高 / 低多少 %
		print('   ', f, '均值 =', round(value, 4), ' 比全表', round(pct, 1), '%')

# 名字写进报告：先有数字，后有名字，顺序不能反（格式：簇X + 最突出的特征 + 需求档位）
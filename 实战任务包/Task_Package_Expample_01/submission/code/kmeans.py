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
# 二维投影说明：本图只是把 4 维特征(temp/hum/windspeed/cnt)压到 temp×cnt 两维画出来，
# 未投影的维度会掩盖簇的真实边界，不能当作完整分群证据
plt.scatter(df['temp'], df['cnt'], c=labels, cmap="tab20", alpha=0.5)
plt.xlabel('temp')
plt.ylabel('cnt')
plt.title('K=4 clusters: temp vs cnt (2D projection only)')   # 标题写明是二维投影（英文避免中文字体豆腐块）
plt.savefig('../figures/cluster_scatter.png', dpi=150)
plt.close()

print(
	'肘部法则：inertia 逐 K 降幅 = [396.3, 202.2, 175.6, 116.7, 101.4, 70.2]（依次对应 K=3…8），降幅到 K=4 还有 202.2，**K=5 起跌破 200 并一路收窄**（175.6→116.7→101.4→70.2）→ 拐点落在 K=4；')
print(
	'轮廓系数：全局最大在 K=2（0.301661），之后一路走低到 K=6（0.225555），K=7、K=8 只微升到 0.229 上下；**K=4（0.264953）高于 K=5~8 的全部取值**（0.237236／0.225555／0.229247／0.229450），也就是 K 再往上分，簇只会更碎；')
print(
	'DBI：最小在 K=8（1.172940）；**K=4（1.254871）优于紧邻的 K=3（1.297733）与 K=5（1.289737）**，在七档里排第 4（中上游），不在差区；')
print('三指标不一致时的通行做法：先用肘部定档（K=4 拐点），再用轮廓系数与 DBI 检验该档不在差区 → 取 **K=4**。')

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

for c in range(4):  # c = 0,1,2,3 逐个簇算
	mask = (labels == c)  # 731 个 True/False，标出哪些天属于簇 c
	cluster_list.append(c)
	temp_list.append(round(df.loc[mask, 'temp'].mean(), 6))  # 该簇 temp 均值
	hum_list.append(round(df.loc[mask, 'hum'].mean(), 6))  # 该簇 hum 均值
	wind_list.append(round(df.loc[mask, 'windspeed'].mean(), 6))  # 该簇 windspeed 均值
	cnt_list.append(round(df.loc[mask, 'cnt'].mean(), 6))  # 该簇 cnt 均值
	count_list.append(int(mask.sum()))  # True 当 1 加起来 = 该簇天数

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
base_temp = df['temp'].mean()  # 基线 = 全表均值（“平均水平”长什么样）
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
		value = df.loc[mask, f].mean()  # 这个簇该特征的均值
		base = df[f].mean()  # 全表该特征的均值
		pct = (value / base - 1) * 100  # 比基线高 / 低多少 %
		print('   ', f, '均值 =', round(value, 4), ' 比全表', round(pct, 1), '%')

# 名字写进报告：先有数字，后有名字，顺序不能反（格式：簇X + 最突出的特征 + 需求档位）
print('簇0 hum最低 windspeed最高 cnt中上 干爽大风中需求日')
print('簇1 temp最低 cnt次低 寒冷低需求日')
print('簇2 temp最高 cnt最高 高需求温暖日')
print('簇3 hum最高 cnt最低 闷湿低需求日')

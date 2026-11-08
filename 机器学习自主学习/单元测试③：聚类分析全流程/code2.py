import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

df = pd.read_csv('city.csv', encoding='utf-8-sig')
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
print(X_scaled.shape)

sils = []
dbis = []

for k in range(2, 11):
	km = KMeans(n_clusters=k, random_state=13, n_init=10)
	km.fit(X_scaled)
	labels = km.labels_
	sils_score = silhouette_score(X_scaled, labels)
	dbi_score = davies_bouldin_score(X_scaled, labels)
	sils.append(sils_score)
	dbis.append(dbi_score)

Ks = list(range(2, 11))
for k, s, d in zip(Ks, sils, dbis):
	print(f'K = {k:>2} 轮廓系数 = {round(s, 4):<6} DBI = {round(d, 4)}')

best_sil = Ks[sils.index(max(sils))]
best_dbi = Ks[dbis.index(min(dbis))]  # 修正：DBI 越小越好，用最小值对应的 K
print(f'轮廓系数峰值K={best_sil}')
print(f'DBI谷值K={best_dbi}')  # 修正：打印 DBI 自己选出的结果，不复用 best_sil

km = KMeans(n_clusters=best_sil, random_state=13, n_init=10)
km.fit(X_scaled)
df['簇标签'] = km.labels_
print(df['簇标签'][:10])
pds = pd.Series(km.labels_).value_counts().sort_index().to_dict()
print(pds)

plt.scatter(df['人口密度'], df['平均房价'], c=df['簇标签'])
plt.xlabel('人口密度')
plt.ylabel('平均房价')
plt.title('人口密度-平均房价散点图')
plt.savefig('人口密度-平均房价散点图.png', dpi=150)
plt.close()

plt.imshow(km.cluster_centers_)
plt.colorbar()
plt.xticks([0, 1, 2], ['人口密度', '平均房价', '地铁站数'])
plt.yticks([0, 1, 2, 3], ['簇0', '簇1', '簇2', '簇3', ])
plt.title('特征和簇热力图')
plt.savefig('特征和簇热力图.png', dpi=150)
plt.close()

means = df.groupby('簇标签').mean().round(2)
print(means)
print('城镇/区县/农村，高价房区/中价房区/低价房区，市中心/镇中心/郊区')
print('簇0：区县市中心中价房区。129.36人口较稀疏，23727.56房价中等，44.89交通最便利')
print('簇1：城镇郊区低价房区。510.95人口最密，7159.27房价最便宜，7.66交通相对不便利')
print('簇2：农村郊区高价房区。57.54人口最稀疏，61684.90房价最贵，2.92交通最不便利')
print('簇3：城镇市中心镇中心。407.84人口次密，30466.42房价中等，20.20交通相对便利')

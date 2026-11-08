import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

df = pd.read_csv('member.csv', encoding='utf-8-sig')
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
print(X_scaled.shape)
print(X_scaled.mean(axis=0).round(4))
print(X_scaled.std(axis=0).round(4))

td = 0
inertias = []
for k in range(2, 11):
	km = KMeans(n_clusters=k, random_state=13, n_init=10)
	km.fit(X_scaled)
	if k == 2:  # 修正：首档没有上一档，按题面用 0 表示
		drop = 0  # 首档下降幅度不做相减
	else:  # 其余档位按上一档减本档计算
		drop = round(td - km.inertia_, 2)  # 计算相邻 K 的 inertia 下降幅度
	print(f'K ={k:>2} inertia={round(km.inertia_, 2):<7} 下降幅度={drop}')  # 输出本档 inertia 和下降幅度
	td = km.inertia_
	inertias.append(km.inertia_)

Ks = list(range(2, 11))
plt.plot(Ks, inertias, 'o-')
plt.xlabel('K簇')
plt.ylabel('Inertia')
plt.title('K - Inertia')
plt.savefig('簇inertia.png', dpi=150)
print("肘部是 K = 3，对比K=2到3，K从3到4下降幅度从59.95到5.72，后续下降幅度一直是1左右，所以K=3为最佳K")

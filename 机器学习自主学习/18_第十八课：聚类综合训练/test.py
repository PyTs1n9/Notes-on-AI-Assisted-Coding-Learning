import pandas as pd
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.compose import ColumnTransformer

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

df = pd.read_csv('day_clean.csv', encoding='utf-8-sig')
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

num_col = ['temp', 'atemp', 'hum', 'windspeed']
cat_col = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']

X = df[num_col + cat_col]
X_train, X_test = train_test_split(
	X, test_size=0.1, random_state=42
)

prep = ColumnTransformer([
	('num', StandardScaler(), num_col),
	('cat', OneHotEncoder(handle_unknown="ignore"), cat_col),
])

X_train_ready = prep.fit_transform(X_train)
X_test_ready = prep.transform(X_test)
print(X_train_ready.shape)
print(X_test_ready.shape)

Ks = list(range(2, 9))
sils = []
dbis = []
inertia = []

for k in Ks:
	km = KMeans(n_clusters=k, random_state=42, n_init=10)
	km.fit(X_train_ready)
	inertia.append(km.inertia_)
	sil = silhouette_score(X_train_ready, km.labels_)
	dbi = davies_bouldin_score(X_train_ready, km.labels_)
	sils.append(sil)
	dbis.append(dbi)
	print(f"K={k}, inertia={round(km.inertia_, 2)}, silhouette={round(sil, 2)}, DBI={round(dbi, 2)}")

best_sil = Ks[sils.index(max(sils))]
best_dbi = Ks[dbis.index(min(dbis))]
print(f'best_sil {best_sil}')
print(f'best_dbi {best_dbi}')

plt.plot(Ks, inertia, 'o-')
plt.xlabel('K簇')
plt.ylabel('Inertia')
plt.title('K簇-Inertia')
plt.savefig('K簇-Inertia图.png', dpi=150)
plt.close()

print('肘部K=2，轮廓系数K=2，DBI K=2，一致')
K = 2
km = KMeans(n_clusters=2, random_state=42, n_init=10)
km.fit(X_train_ready)
train_result = df.loc[X_train.index].copy()  # 按原始索引取回训练行
train_result['簇标签'] = km.labels_  # 添加训练集聚类标签

plt.scatter(train_result['temp'], train_result['hum'], c=train_result['簇标签'])  # 函数⑨：按簇着色
plt.xlabel('temp')  # 设置横轴字段
plt.ylabel('hum')  # 设置纵轴字段
plt.title(f'Training clusters, K={best_sil}')  # 设置标题并标明 K
plt.savefig('day_cluster_scatter.png', dpi=150)  # 保存散点图
plt.close()  # 关闭当前图，释放绘图资源

plt.imshow(km.cluster_centers_)  # 函数⑨：画标准化/独热后的中心矩阵
plt.colorbar()  # 显示颜色对应的数值
plt.xlabel('Processed feature index')  # 横轴代表处理后的特征索引
plt.ylabel('Cluster')  # 纵轴代表簇
plt.title('KMeans centers in processed feature space')  # 标明热力图展示的是处理后空间
plt.savefig('day_cluster_centers.png', dpi=150)  # 保存中心热力图
plt.close()  # 关闭当前图
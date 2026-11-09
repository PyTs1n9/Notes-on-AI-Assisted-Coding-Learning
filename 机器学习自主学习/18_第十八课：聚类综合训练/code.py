import pandas as pd  # 导入 pandas，用于读表和整理结果
import matplotlib.pyplot as plt  # 导入绘图库
from sklearn.cluster import KMeans  # 导入 KMeans 聚类模型
from sklearn.compose import ColumnTransformer  # 函数③：按列选择处理流程
from sklearn.model_selection import train_test_split  # 函数①：划分训练数据和测试数据
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 函数②/④：类别编码与数值标准化
from sklearn.metrics import silhouette_score, davies_bouldin_score  # 函数⑥/⑦：聚类评价

raw_df = pd.read_csv('day_clean.csv', encoding='utf-8-sig')  # 读取 CSV，正确处理 UTF-8 BOM
categorical_features = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']  # 列出类别字段
numeric_features = ['temp', 'atemp', 'hum', 'windspeed']  # 列出连续数值字段
feature_columns = numeric_features + categorical_features  # 合并特征字段名
X = raw_df[feature_columns]  # 从原表取出聚类特征

X_train, X_test = train_test_split(X, test_size=0.25, random_state=21)  # 函数①：固定划分结果
# 创建分列预处理器：数值列标准化，类别列独热编码
preprocessor = ColumnTransformer([  # 函数③：建立两类特征处理规则
	('numeric', StandardScaler(), numeric_features),  # 函数④：数值列按训练集统计量标准化
	('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)  # 函数②：类别列编码
])  # 结束列转换器定义

X_train_ready = preprocessor.fit_transform(X_train)  # 函数③：训练集 fit 并转换
X_test_ready = preprocessor.transform(X_test)  # 函数③：测试集只 transform

Ks = list(range(2, 9))  # 创建 K=2 到 K=8 的候选值
inertias = []  # 保存每个 K 的 inertia
silhouettes = []  # 保存每个 K 的轮廓系数
dbis = []  # 保存每个 K 的 DBI
# 逐个候选 K 拟合模型并记录训练集指标
for k in Ks:  # 依次检查候选簇数
	km = KMeans(n_clusters=k, random_state=21, n_init=10)  # 函数⑤：创建当前 K 的模型
	km.fit(X_train_ready)  # 函数⑤：只在训练特征上拟合
	inertias.append(km.inertia_)  # 保存簇内距离平方和
	silhouettes.append(silhouette_score(X_train_ready, km.labels_))  # 函数⑥：计算训练集轮廓系数
	dbis.append(davies_bouldin_score(X_train_ready, km.labels_))  # 函数⑦：计算训练集 DBI
	print(f'K={k}, inertia={km.inertia_:.2f}, silhouette={silhouettes[-1]:.4f}, DBI={dbis[-1]:.4f}')  # 打印当前 K 的实测指标

best_sil_k = Ks[silhouettes.index(max(silhouettes))]  # 轮廓系数取最大值对应 K
best_dbi_k = Ks[dbis.index(min(dbis))]  # DBI 取最小值对应 K
print('silhouette best K:', best_sil_k)  # 输出轮廓系数峰值 K
print('DBI best K:', best_dbi_k)  # 输出 DBI 谷值 K

final_model = KMeans(n_clusters=best_sil_k, random_state=21, n_init=10)  # 函数⑤：创建最终模型
final_model.fit(X_train_ready)  # 函数⑤：拟合最终聚类
test_labels = final_model.predict(X_test_ready)  # 使用已训练模型预测留出样本的簇

train_result = raw_df.loc[X_train.index].copy()  # 按原始索引取回训练行
train_result['簇标签'] = final_model.labels_  # 添加训练集聚类标签

print('train transformed shape:', X_train_ready.shape)  # 输出训练处理矩阵形状
print('test transformed shape:', X_test_ready.shape)  # 输出测试处理矩阵形状
print('test cluster counts:', pd.Series(test_labels).value_counts().sort_index().to_dict())  # 输出测试样本各簇数量
print('train cluster counts:', pd.Series(final_model.labels_).value_counts().sort_index().to_dict())  # 输出训练样本各簇数量
print('business profile:')  # 打印画像表标题
print(train_result.groupby('簇标签')[numeric_features + ['cnt']].mean().round(2))  # 函数⑧：按簇比较原始单位均值

plt.scatter(train_result['temp'], train_result['hum'], c=train_result['簇标签'])  # 函数⑨：按簇着色
plt.xlabel('temp')  # 设置横轴字段
plt.ylabel('hum')  # 设置纵轴字段
plt.title(f'Training clusters, K={best_sil_k}')  # 设置标题并标明 K
plt.savefig('day_cluster_scatter.png', dpi=150)  # 保存散点图
plt.close()  # 关闭当前图，释放绘图资源

plt.imshow(final_model.cluster_centers_)  # 函数⑨：画标准化/独热后的中心矩阵
plt.colorbar()  # 显示颜色对应的数值
plt.xlabel('Processed feature index')  # 横轴代表处理后的特征索引
plt.ylabel('Cluster')  # 纵轴代表簇
plt.title('KMeans centers in processed feature space')  # 标明热力图展示的是处理后空间
plt.savefig('day_cluster_centers.png', dpi=150)  # 保存中心热力图
plt.close()  # 关闭当前图

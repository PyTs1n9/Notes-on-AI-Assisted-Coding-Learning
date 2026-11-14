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

print('轮廓系数峰值 K：', best_sil)  # 打印程序从 sils 最大值映射出的 K
print('DBI 谷值 K：', best_dbi)  # 打印程序从 dbis 最小值映射出的 K
print('两个指标是否一致：', best_sil == best_dbi)  # 比较两个动态结果
K = best_sil  # 本轮按轮廓系数峰值选择最终 K
print('最终选择 K：', K, '；本轮未独立确认肘部，采用轮廓系数候选。')  # 不把左端候选 K 冒充肘部结论
km = KMeans(n_clusters=K, random_state=42, n_init=10)
km.fit(X_train_ready)
pred = km.predict(X_test_ready)
train_result = df.loc[X_train.index].copy()  # 按原始索引取回训练行
train_result['簇标签'] = km.labels_  # 添加训练集聚类标签

plt.scatter(train_result['temp'], train_result['hum'], c=train_result['簇标签'])  # 函数⑨：按簇着色
plt.xlabel('temp')  # 设置横轴字段
plt.ylabel('hum')  # 设置纵轴字段
plt.title(f'Training clusters, K={best_sil}')  # 设置标题并标明 K
plt.savefig('day_cluster_scatter.png', dpi=150)  # 保存散点图
print('--- 错误点参考答案运行结果 ---')  # 区分原程序输出与老师修正部分
print('轮廓系数峰值 K：', best_sil)  # 使用原列表实际计算出的候选值
print('DBI 谷值 K：', best_dbi)  # 使用 DBI 最小值对应的候选值
print('两个指标是否一致：', best_sil == best_dbi)  # 不再手写两个指标一致
K = best_sil  # 本参考按轮廓系数峰值选最终 K，同时保留 DBI 比较结果
print('最终选择 K：', K, '；本轮未独立确认肘部，采用轮廓系数候选。')  # 不把候选左端点当成已验证拐点
km = KMeans(n_clusters=K, random_state=42, n_init=10)  # 按最终 K 创建模型
km.fit(X_train_ready)  # 只在训练矩阵上拟合最终模型
test_labels = km.predict(X_test_ready)  # 给测试样本分配簇号，不再拟合模型
print('测试样本簇号：', test_labels)  # 输出每个测试样本的实际簇号
print('测试样本簇号数量：', len(test_labels))  # 核查每个测试样本都有一个簇号
train_result = df.loc[X_train.index].copy()  # 按训练矩阵的行序取回原表，包括 cnt
train_result['簇标签'] = km.labels_  # 将最终模型的训练簇号按相同顺序贴回
cluster_groups = train_result.groupby('簇标签')  # 先按训练簇号分组
profile = cluster_groups[num_col + ['cnt']].mean()  # 计算连续字段与 cnt 的原始单位均值
profile = profile.round(2)  # 将画像均值保留两位小数便于阅读
print('训练簇画像：')  # 打印画像标题
print(profile)  # 给下面的解释提供可核查的数字
train_temp_mean = train_result['temp'].mean()  # 取得全部训练样本的温度均值作比较基准
train_cnt_mean = train_result['cnt'].mean()  # 取得全部训练样本的租车量均值作比较基准
for cluster_id in range(K):  # 为每个簇输出一条基于本次数据的解释
	temp_mean = profile.loc[cluster_id, 'temp']  # 读取当前簇的温度均值
	hum_mean = profile.loc[cluster_id, 'hum']  # 读取当前簇的湿度均值
	cnt_mean = profile.loc[cluster_id, 'cnt']  # 读取当前簇的租车量均值
	if temp_mean < train_temp_mean:  # 与全部训练样本温度均值比较
		temp_text = '相对较凉爽'  # 不把归一化温度直接叫作摄氏度
	else:  # 当前簇温度不低于训练样本总体均值
		temp_text = '相对较温暖'  # 温暖描述只表示本批训练数据中的相对位置
	if cnt_mean < train_cnt_mean:  # 与全部训练样本租车量均值比较
		cnt_text = '平均租车量相对较低'  # 只描述统计现象，不推断因果
	else:  # 当前簇平均租车量不低于训练样本总体均值
		cnt_text = '平均租车量相对较高'  # 只描述本次分组结果
	print('簇', cluster_id, '：', temp_text, '；temp 均值', temp_mean, '；hum 均值', hum_mean, '；cnt 均值', cnt_mean, '，',
	      cnt_text)  # 数字全部取自画像表
print('说明：这些数字是分组后的统计关联，不能据此声称温度导致租车量变化。')  # 不把画像解释写成因果结论
plt.scatter(train_result['temp'], train_result['hum'], c=train_result['簇标签'])  # 用最终训练簇号给散点着色
plt.xlabel('temp')  # 横轴使用原表温度字段
plt.ylabel('hum')  # 纵轴使用原表湿度字段
plt.title(f'Training clusters, K={K}')  # 标明真正使用的最终 K
plt.savefig('day_cluster_scatter.png', dpi=150)  # 按题目文件名保存散点图
plt.close()  # 关闭散点图
plt.imshow(km.cluster_centers_)  # 展示最终模型在处理后空间中的簇中心
plt.colorbar()  # 显示颜色标尺
plt.xlabel('Processed feature index')  # 横轴说明是处理后的特征索引
plt.ylabel('Cluster')  # 纵轴表示簇号
plt.title('KMeans centers in processed feature space')  # 不把处理后中心冒充原始字段单位
plt.savefig('day_cluster_centers.png', dpi=150)  # 按题目文件名保存中心热力图
plt.close()  # 关闭热力图

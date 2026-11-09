### 一、概念题（每题 2 分，共 10 分）

1. 什么是无监督学习？KMeans 的训练数据里有没有标签 y？它靠什么判断两个样本「像」？
无监督学习就是让机器在没有标签的数据中发现规律。
KMeans 的训练数据里没有标签y。
靠样本到样本簇中心的距离。

2. 为什么 KMeans 之前必须 `StandardScaler` 标准化？不做的后果是什么？
确保特征数据都是统一单位。
如果不做，就会影响最终标签的预测，比如火车票和行驶距离，就必须标准化。

3. `inertia_` 是什么（说人话）？为什么「选 inertia 最小的 K」是错的？肘部法则怎么定 K？
inertia是每个簇中样本到簇中心的平均距离。
因为随K增大，inertia是必然减小的，簇数量的增多让样本的相关性被增强了，离散的样本少了，距离就少了，所以不能直接选inertia最小的K。
定inertia最能体现由陡到缓的那个K。

4. 轮廓系数和 DBI 各是越大越好还是越小越好？`silhouette_score(X, labels)`、`davies_bouldin_score(X, labels)` 里的 `labels` 从哪来、有什么讲究？
轮廓系数越大越好，DBI越小越好。
从KMeans模型中得来，labels = km.labels_，这个是每个样本的簇号。

5. 散点图、聚类中心热力图、分组平均特征表三件套各回答什么问题？给簇起名字时为什么必须引自己表里的数字？
散点图是每个样本的分散程度，用来回答这个簇分的好不好；
聚类中心热力图是和K簇的相关性，用来看这个K某个指数高还是低；
分组平均特征表不知道。
因为需要通过数字大小来判断这个指数是高还是低。


### 二、函数题（每题 2 分，共 8 分）

1. `StandardScaler().fit_transform(X)` 返回什么？标准化后各列的均值、标准差约等于多少？
返回被标准化后的特征。
均值约等于0.。
标准差约等于1.。

2. `KMeans(n_clusters=3, random_state=13, n_init=10)` 三个参数各管什么？`km.labels_`、`km.cluster_centers_`、`km.inertia_` 三张成绩单各是什么、什么时候才有？
分别管K的数量，随机种子，分10次的最小值。
每个样本的簇号，簇中心，簇内样本到簇中心距离的平方和，聚类的时候才有。

3. `silhouette_score(X, labels)` 和 `davies_bouldin_score(X, labels)` 的返回值各是什么、方向如何？为什么说 `labels` 必须是「同一次聚类」给出的 `km.labels_`？
返回的是特征的轮廓系数，返回的是特征的DBI指数，方向不知道。
labels是样本的簇号，不同簇，编号就不同。

4. `plt.plot(Ks, inertias, 'o-')` 的 x、y 各传什么？`plt.scatter(x, y, c=...)` 的 `c` 管什么？`plt.imshow(km.cluster_centers_)` 画的是什么、默认横纵轴各代表什么？
每个簇号，样本到该簇的平均距离。
簇标签数组。
热力图，横轴是参数，纵轴是K簇。


### 三、纠错题（每题 3 分，共 12 分：指出错误 + 写出正确写法/说法）

1. 有人直接 `km = KMeans(n_clusters=3).fit(df)`，在原始数据上聚类——错在哪？会出什么后果？正确写法？
错在没有标准化，会导致样本到簇中心有的很近有的很远。
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
km = KMeans(n_clusters=3).fit(df_scaled)

2. `s = silhouette_score(km.labels_, X_scaled)` ——错在哪？正确写法？
样本簇号和特征写反了
s = silhouette_score(X_scaled, km.labels_)
先数据，再簇号

3. 有人总结：「K=10 的 `inertia_` 最小，所以最佳 K 就取 10」——错在哪？正确的定 K 方法是什么？
错在inertia_最小就是最佳K，正确的K应该是通过画图或者统计inertia数值，比较出inertia从陡到缓的节点。

4. 解读只写「簇 0 是高消费群体」，不带任何数字——错在哪？举一个正确的写法。
错在没有依据。
K=0 ,temp最高，windspeed最低，说明这天是高温无风天气。


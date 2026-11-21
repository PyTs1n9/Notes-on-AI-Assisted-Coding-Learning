# A04：参考实现。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "A_参考案例全覆盖", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "A04")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scores = pd.read_csv(os.path.join(DATA, 'scores_new.csv'))
# 所有行保留，从第二列取特征；不要把编号标准化后混入距离。
features = scores.iloc[:, 1:]
# 标准化先于 KMeans 和 PCA；两者都接收相同的标准化分数。
scaled = StandardScaler().fit_transform(features)
# 右边界不含 7，才能覆盖题设的全部 K。
k_values = list(range(2, 7))
sse = []
for k in k_values:
    model = KMeans(n_clusters=k, random_state=34, n_init=10)
    model.fit(scaled)
    # 拟合后的 inertia_ 是每个样本距其簇中心的平方距离之和。
    sse.append(model.inertia_)
model = KMeans(n_clusters=3, random_state=34, n_init=10)
labels = model.fit_predict(scaled)
pca = PCA(n_components=2)
# 这里投影用于展示；先对全部标准化数据拟合 PCA，再得到二维坐标。
projection = pca.fit_transform(scaled)
view = pd.DataFrame({'id': scores['student_id'], 'PC1': projection[:, 0], 'PC2': projection[:, 1], 'cluster': labels})
profile = features.assign(cluster=labels).groupby('cluster').mean()
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].plot(k_values, sse, marker='o')
axes[0].set(xlabel='K', ylabel='Inertia')
# 颜色的字段是聚类标签，不是编号、分数或二维坐标。
sns.scatterplot(data=view, x='PC1', y='PC2', hue='cluster', palette='Set2', ax=axes[1])
sns.heatmap(profile, annot=True, fmt='.1f', cmap='Blues', ax=axes[2])
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'score_clusters.png'), dpi=120)
plt.close(fig)
pd.DataFrame({'k': k_values, 'inertia': sse}).to_csv(os.path.join(OUT, 'inertia.csv'), index=False)
view.to_csv(os.path.join(OUT, 'score_clusters.csv'), index=False)
profile.to_csv(os.path.join(OUT, 'profile.csv'))
pd.DataFrame({'component': ['PC1', 'PC2'], 'ratio': pca.explained_variance_ratio_}).to_csv(os.path.join(OUT, 'variance.csv'), index=False)
print('PCA 解释方差比例', pca.explained_variance_ratio_.round(4).tolist())
print('簇数', view['cluster'].nunique())

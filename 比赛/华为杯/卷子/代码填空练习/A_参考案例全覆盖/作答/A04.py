# A04：只替换带题号的空位。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "A04")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scores = pd.read_csv(os.path.join(DATA, 'scores_new.csv'))
features = scores.iloc[__A04_01__]
scaled = __A04_02__
k_values = list(range(__A04_03__))
sse = []
for k in k_values:
    model = KMeans(n_clusters=k, random_state=34, n_init=10)
    model.fit(scaled)
    sse.append(__A04_04__)
model = KMeans(n_clusters=3, random_state=34, n_init=10)
labels = model.fit_predict(scaled)
pca = PCA(n_components=2)
projection = __A04_05__
view = pd.DataFrame({'id': scores['student_id'], 'PC1': projection[:, 0], 'PC2': projection[:, 1], 'cluster': labels})
profile = features.assign(cluster=labels).groupby('cluster').mean()
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].plot(k_values, sse, marker='o')
axes[0].set(xlabel='K', ylabel='Inertia')
sns.scatterplot(data=view, x='PC1', y='PC2', hue=__A04_06__, palette='Set2', ax=axes[1])
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

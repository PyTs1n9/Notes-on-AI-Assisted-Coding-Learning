# A03：只替换带题号的空位。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "A03")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

frame = pd.read_csv(os.path.join(DATA, 'orders_data.csv'))
valid = frame.loc[frame['visits'] >= 0].copy()
valid['spend'] = valid['spend'].fillna(__A03_01__)
encoder = LabelEncoder()
valid['tier_code'] = encoder.fit_transform(__A03_02__)
features = valid[['visits', 'spend', 'tier_code']]
scaler = StandardScaler()
scaled = scaler.__A03_03__(features)
candidates = list(range(2, 6))
scores = []
for k in candidates:
    km = KMeans(n_clusters=k, random_state=29, n_init=10)
    labels = km.fit_predict(scaled)
    scores.append(silhouette_score(__A03_04__))
comparison = pd.DataFrame({'k': candidates, 'silhouette': scores})
best_k = int(comparison.loc[__A03_05__, 'k'])
chosen = KMeans(n_clusters=best_k, random_state=29, n_init=10)
valid['cluster'] = chosen.fit_predict(scaled)
profile = valid.groupby('cluster')[['visits', 'spend']].mean().reset_index()
valid[['id', 'cluster']].to_csv(os.path.join(OUT, 'user_clusters.csv'), index=False)
comparison.to_csv(os.path.join(OUT, 'k_scores.csv'), index=False)
profile.to_csv(os.path.join(OUT, 'cluster_profile.csv'), index=False)
plt.plot(candidates, scores, marker='o')
plt.xlabel('K')
plt.ylabel('Silhouette score')
plt.savefig(os.path.join(OUT, 'silhouette.png'), dpi=120)
plt.close()
print('选定 K', best_k)
print('各簇样本数', valid['cluster'].value_counts().sort_index().to_dict())

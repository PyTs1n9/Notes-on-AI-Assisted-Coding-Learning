# A03：参考实现。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "A_参考案例全覆盖", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "A03")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

frame = pd.read_csv(os.path.join(DATA, 'orders_data.csv'))
valid = frame.loc[frame['visits'] >= 0].copy()
# 先剔除不合法的访问记录，统计量以保留样本为准。
valid['spend'] = valid['spend'].fillna(valid['spend'].median())
encoder = LabelEncoder()
# 编码的是按前缀排序的单个等级字段；无序类别一般应独热编码。
valid['tier_code'] = encoder.fit_transform(valid['tier'])
features = valid[['visits', 'spend', 'tier_code']]
scaler = StandardScaler()
# 一次拟合所选列的尺度并完成转换；不把用户编号当特征。
scaled = scaler.fit_transform(features)
candidates = list(range(2, 6))
scores = []
for k in candidates:
    km = KMeans(n_clusters=k, random_state=29, n_init=10)
    labels = km.fit_predict(scaled)
    # 轮廓分数用聚类所在的标准化空间，第二个参数是该 K 的标签。
    scores.append(silhouette_score(scaled, labels))
comparison = pd.DataFrame({'k': candidates, 'silhouette': scores})
# 索引取最大值首次出现的位置，配合递增 K 实现并列取小。
best_k = int(comparison.loc[comparison['silhouette'].idxmax(), 'k'])
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

# B02：参考实现。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "B_客服运营与设备巡检", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "B02")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

members = pd.read_csv(os.path.join(DATA, 'member_usage.csv'))
# 缺失渠道单独作为类别，不用数字 0 充当渠道名。
members['channel'] = members['channel'].fillna('未知')
members['monthly_visits'] = members['monthly_visits'].fillna(members['monthly_visits'].median())
numbers = members[['monthly_visits', 'session_min']]
# 已填补的两个数值特征才参与标准化；ID 和独热列不传入。
scaled = StandardScaler().fit_transform(numbers)
# 每类一列，不强加 0、1、2 之间的距离。
channels = pd.get_dummies(members['channel'], prefix='channel', dtype=int)
scaled_frame = pd.DataFrame(scaled, index=members.index, columns=numbers.columns)
# 按列拼接，保留每个用户的同一行。
features = pd.concat([scaled_frame, channels], axis=1, sort=False)
labels = KMeans(n_clusters=3, n_init=10, random_state=71).fit_predict(features)
# 可视化的 PCA 输入与聚类一致；PCA 的输出不是原始统计单位。
display = PCA(n_components=2).fit_transform(features)
view = pd.DataFrame({'id': members['id'], 'pc1': display[:, 0], 'pc2': display[:, 1], 'cluster': labels})
original_means = members.assign(cluster=labels).groupby('cluster')[['monthly_visits', 'session_min']].mean()
channel_counts = pd.crosstab(view['cluster'], members['channel'])
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.scatterplot(data=view, x='pc1', y='pc2', hue='cluster', palette='deep', ax=axes[0])
chart_counts = channel_counts.reindex(columns=['应用', '小程序', '柜台', '未知']).copy()
chart_counts.columns = ['App', 'Mini app', 'Counter', 'Unknown']
# 热图画各簇各渠道的原始人数，而不是标准化过的坐标。
sns.heatmap(chart_counts, annot=True, fmt='d', ax=axes[1])
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'member_map.png'), dpi=120)
plt.close(fig)
view.to_csv(os.path.join(OUT, 'member_clusters.csv'), index=False)
original_means.to_csv(os.path.join(OUT, 'member_means.csv'))
channel_counts.to_csv(os.path.join(OUT, 'channel_counts.csv'), encoding='utf-8-sig')
print('数值列、独热列', numbers.shape[1], channels.shape[1])
print('簇数、入簇记录数', view['cluster'].nunique(), len(view))

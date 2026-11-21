# C02：只替换带题号的空位。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "C02")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

history = pd.read_csv(os.path.join(DATA, 'history.csv'))
today = pd.read_csv(os.path.join(DATA, 'today.csv'))
cols = ['weekly_visits', 'indoor_min', 'outdoor_min']
scaler = StandardScaler()
old_scaled = scaler.__C02_01__(history[cols])
new_scaled = scaler.__C02_02__(today[cols])
clusterer = KMeans(n_clusters=3, random_state=61, n_init=10)
old_labels = clusterer.fit_predict(old_scaled)
new_labels = clusterer.__C02_03__(new_scaled)
pca = PCA(n_components=2)
old_xy = pca.fit_transform(old_scaled)
new_xy = pca.__C02_04__(new_scaled)
old = pd.DataFrame({'id': history['id'], 'PC1': old_xy[:, 0], 'PC2': old_xy[:, 1], 'cluster': old_labels})
new = pd.DataFrame({'id': today['id'], 'PC1': new_xy[:, 0], 'PC2': new_xy[:, 1], 'cluster': new_labels})
fig, ax = plt.subplots(figsize=(7, 5))
palette = {0: '#4575b4', 1: '#d73027', 2: '#4d9221'}
sns.scatterplot(data=old, x='PC1', y='PC2', hue='cluster', palette=palette, ax=ax)
sns.scatterplot(data=new, x='PC1', y='PC2', hue=__C02_05__, palette=palette, marker='X', s=130, legend=False, ax=ax)
ax.set_title('History: circles / Today: crosses')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'visitor_overlay.png'), dpi=120)
plt.close(fig)
old.to_csv(os.path.join(OUT, 'history_map.csv'), index=False)
new.to_csv(os.path.join(OUT, 'today_map.csv'), index=False)
print('历史坐标、新坐标', old_xy.shape, new_xy.shape)
print('今日访客簇号', new_labels.tolist())

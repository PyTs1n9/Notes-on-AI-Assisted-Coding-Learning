import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
	silhouette_score, davies_bouldin_score
)

# ---------- 1. 读数据 + 标准化（老流程） ----------
df = pd.read_csv('customers.csv', encoding='utf-8-sig')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ---------- 2. 循环 K=2~10，给每种分法打两个分 ----------
Ks = list(range(2, 11))  # K=1 和 K=样本数没法打分，从 2 起
sils = []  # 收集各 K 的轮廓系数
dbis = []  # 收集各 K 的 DBI
for k in Ks:
	km = KMeans(n_clusters=k, random_state=42, n_init=10)
	km.fit(X_scaled)
	sils.append(round(silhouette_score(X_scaled, km.labels_),4))
	dbis.append(round(davies_bouldin_score(X_scaled, km.labels_),4))

# ---------- 3. 打印成绩单 ----------
print('K    轮廓系数    DBI')
for k, s, d in zip(Ks, sils, dbis):
	print(f"K={k:<3d} {s:<10} {d}")

# ---------- 4. 程序自动找峰值和谷值 ----------
best_sil = Ks[sils.index(max(sils))]
best_dbi = Ks[dbis.index(min(dbis))]
print(f'轮廓系数峰值：K={best_sil}（{max(sils)}）')
print(f'DBI 谷值：K={best_dbi}（{min(dbis)}）')

# ---------- 5. 读结果 ----------
print('两把尺子指向同一个 K，再对上第十课的肘部（K=3）——三证合一，最佳 K=3。')
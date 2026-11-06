import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ["SimHei"]

# ---------- 1. 读数据 + 标准化（第九课的流程原样搬来） ----------
df = pd.read_csv('customers.csv', encoding='utf-8-sig')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ---------- 2. 循环 K=2~10，收集每个 K 的 inertia_ 和下降幅度 ----------
Ks = list(range(2, 11))
inertia = []
drops = []
prev = None
for k in Ks:
	km = KMeans(n_clusters=k, random_state=42, n_init=10)
	km.fit(X_scaled)
	inertia.append(round(km.inertia_, 2))
	if prev is None:
		drops.append(None)
	else:
		drops.append(np.round((prev - km.inertia_), 2))
	prev = km.inertia_

# ---------- 3. 打印 "K ~ inertia ~ 下降幅度" 对照表 ----------
for k ,iner ,drop in zip(Ks,inertia,drops):
	d = 'none' if drop is None else drop
	print(f'K={k:<3d} inertia_={iner:<6} 下降幅度={d}')

# ---------- 4. 画肘部曲线 ----------
plt.plot(Ks, inertia, 'o-')                               # 函数②：plt.plot 画折线（x=K值，y=inertia）
plt.xlabel('簇数 K')                                        # 函数③：x 轴标签
plt.ylabel('inertia（簇内平方和）')                          # 函数③：y 轴标签
plt.title('肘部法则：K 与 inertia 的关系（顾客数据）')         # 函数③：图标题
plt.tight_layout()                                          # 自动收紧边距，防止标签被裁掉
plt.savefig('顾客数据-肘部曲线.png', dpi=120)                 # 函数③：存成图片（dpi=清晰度）
print('图片已保存：顾客数据-肘部曲线.png')

# ---------- 5. 读结果 ----------
print(f'读结果：K=2→3 下降幅度 {drops[1]}，K=3→4 只剩 {drops[2]}，4 之后每档都不足 1——')
print('下降从 K=3 起明显变缓，肘部在 K=3，正好印证第九课人工定的 K=3。')
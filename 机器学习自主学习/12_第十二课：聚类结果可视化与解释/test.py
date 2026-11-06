import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_csv('athletes.csv', encoding='utf-8-sig')

print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

km = KMeans(n_clusters=3, random_state=42, n_init=10)
km.fit(X_scaled)

df['簇标签'] = km.labels_

size = df['簇标签'].value_counts().sort_index()
print(size)

means = df.groupby('簇标签').mean().round(2)
print(means)

plt.scatter(df['速度'], df['力量'], c=df['簇标签'], cmap='viridis', s=10)
plt.xlabel('速度')
plt.ylabel('力量')
plt.title('速度 - 力量 散点图')
plt.savefig('速度-力量散点图.png', dpi=150)
plt.close()

plt.imshow(km.cluster_centers_)
plt.colorbar(label='标准化后的值')
plt.xticks([0, 1, 2], ['速度', '力量', '耐力'])
plt.yticks([0, 1, 2], ['簇0', '簇1', '簇2'])
plt.title('聚类中心热力图（标准化值）')
plt.savefig('运动员-中心热力图.png', dpi=150)
plt.close()

print('簇0 耐力型 耐力最高 速度54.76，力量53.44都很低，耐力83.94很高')
print('簇1 均衡型 速度65.11力量65.38耐力65.24都很均衡，不高不低')
print('簇2 爆发型 速度81.04力量78.34最高，耐力52.51很低，爆发好，耐力差')

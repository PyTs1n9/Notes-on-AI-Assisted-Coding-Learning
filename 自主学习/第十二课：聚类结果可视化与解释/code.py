import pandas as pd  # 读数据、分组统计
import matplotlib.pyplot as plt  # 画图
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.cluster import KMeans  # KMeans

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 1. 老流程：读数据 → 标准化 → KMeans(K=3) ----------
df = pd.read_csv('customers.csv', encoding='utf-8-sig')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

km = KMeans(n_clusters=3, random_state=42, n_init=10)
km.fit(X_scaled)

df['簇标签'] = km.labels_
print(df.head())

# ---------- 2. 二维散点图：每个点按簇着色 ----------
plt.figure(figsize=(7, 5))  # 新建画布（宽 7 高 5 英寸）
plt.scatter(df['年收入（元）'], df['消费评分'],  # x、y 各传一列
            c=df['簇标签'], cmap='viridis', s=40)  # 函数①：c=着色依据（簇标签）
plt.xlabel('年收入（元）')  # x 轴标签
plt.ylabel('消费评分')  # y 轴标签
plt.title('顾客 3 簇散点图（颜色 = 簇号）')  # 标题
plt.savefig('顾客-散点图.png', dpi=120)  # 存成图片（dpi=清晰度）
plt.close()  # 关掉画布，防止和下一张图串台

# ---------- 3. 聚类中心热力图 ----------
plt.figure(figsize=(5, 4))
plt.imshow(km.cluster_centers_, cmap='coolwarm', aspect='auto')
plt.colorbar(label='标准化后的值')
plt.xticks([0, 1], ['年收入', '消费评分'])                 # x 轴刻度换成特征名（位置、标签成对传）
plt.yticks([0, 1, 2], ['簇0', '簇1', '簇2'])              # y 轴刻度换成簇号
plt.title('聚类中心热力图（标准化值）')
plt.savefig('顾客-中心热力图.png', dpi=120)                # 存成图片
plt.close()

# ---------- 4. 统计不同簇的平均特征（原始尺度，最好读） ----------
means = df.groupby('簇标签').mean().round(2)
sizes = df['簇标签'].value_counts()
print(sizes)
sizes = df['簇标签'].value_counts().sort_index()
print(sizes)
print(means)

# ---------- 5. 读结果：数字用 f-string 从表里拼，不手写 ----------
for i in range(3):                                       # 逐簇读
    print(f'簇{i}：{sizes[i]} 人，平均年收入 {means.loc[i, "年收入（元）"]:.0f} 元，'
          f'平均消费评分 {means.loc[i, "消费评分"]:.1f}')
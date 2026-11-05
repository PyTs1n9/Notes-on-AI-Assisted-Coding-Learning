import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# Step 1：加载数据并合并特征与标签
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df = pd.DataFrame(df)
df.to_csv('2-6.csv', encoding='utf-8-sig', index=False)

# 以下为题目提供的脏数据生成代码，不得修改
rng = np.random.RandomState(42)
n = len(df)
df.loc[rng.rand(n) < 0.04, 'mean texture'] = np.nan  # 约 4% 缺失值
df.loc[rng.choice(n, 8, replace=False), 'mean radius'] = -1.0  # 8 个非法值
df.loc[rng.choice(n, 6, replace=False), 'mean perimeter'] *= 100  # 6 个极端值
df = pd.concat([df, df.sample(12, random_state=0)], ignore_index=True)  # 12 条重复行

# Step 2
print(df.dtypes)
print(df.describe())

plt.hist(df['mean radius'])
plt.xlabel('分数')
plt.ylabel('个数')
plt.title('mean radius直方图')
plt.savefig('mean radius直方图.png', dpi=150)
plt.close()

plt.scatter(df['mean radius'], df['target'], alpha=0.5)
plt.xlabel('mean radius')
plt.ylabel('target')
plt.title('mean radius - target 散点图')
plt.savefig('mean radius和target散点图.png', dpi=150)
plt.close()

print('重复值', df.duplicated().sum())
print('缺失值', df.isnull().sum())
# Q1 = df['mean perimeter'].quantile(0.25)
# Q3 = df['mean perimeter'].quantile(0.75)
# IQR = Q3 - Q1
# lower = Q1 - 1.5 * IQR
# upper = Q3 + 1.5 * IQR
# out = df[(df['mean perimeter'] < lower) | (df['mean perimeter'] > upper)]
print('非法值', (df['mean radius'] == -1.0).sum())
print('异常值', 6)

df = df.drop_duplicates()
for i in range(len(df['mean radius'])):
	df.loc[i, -1.0] = np.nan

X = df.drop(columns='target')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

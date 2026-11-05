import numpy as np
import pandas as pd
from IPython.core.pylabtools import figsize
from sklearn.datasets import load_breast_cancer

# Step 1：加载数据并合并特征与标签
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
print(df.shape)

# 以下为题目提供的脏数据生成代码，不得修改
rng = np.random.RandomState(42)
n = len(df)
df.loc[rng.rand(n) < 0.04, 'mean texture'] = np.nan  # 约 4% 缺失值
df.loc[rng.choice(n, 8, replace=False), 'mean radius'] = -1.0  # 8 个非法值
df.loc[rng.choice(n, 6, replace=False), 'mean perimeter'] *= 100  # 6 个极端值
df = pd.concat([df, df.sample(12, random_state=0)], ignore_index=True)  # 12 条重复行

# Step 2 EDA 与诊断

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for label, name, color in ((0, 'malignant(0)', 'blue'), (1, 'benign(1)', 'red')):
	axes[0].hist(df.loc[df['target'] == label, 'mean radius'], label=name, color=color, alpha=0.6)
axes[0].set_title('malignant - benign hist')
axes[0].set_xlabel('mean radius')
axes[0].set_ylabel('count')
axes[0].legend()

axes[1].scatter(df['mean radius'], df['mean texture'], c=df['target'], alpha=0.5, cmap='coolwarm')
axes[1].set_title('mean radius vs mean texture')
axes[1].set_xlabel('mean radius')
axes[1].set_ylabel('mean texture')
plt.tight_layout()
plt.savefig('2-6 img.png', dpi=150)

# 诊断重复、非法值、缺失值和异常值
print('重复值：', df.duplicated().sum())
print('非法值：', (df['mean radius'] < 0).sum())
print('缺失值：', df.isnull().sum().sum())
print('极端值：', (df['mean perimeter'] > 300).sum())

# Step 3 清洗与预处理
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

df = df.drop_duplicates().reset_index(drop=True)
df.loc[df['mean radius'] < 0, 'mean radius'] = np.nan
df.loc[df['mean perimeter'] > 300, 'mean perimeter'] = np.nan
feature_cols = [c for c in df.columns if c not in 'target']
pre = ColumnTransformer([
	('num', Pipeline([
		('impute', SimpleImputer(strategy='median')),
		('scaler', StandardScaler())
	]), feature_cols)
])

# Step 4 划分数据集
from sklearn.model_selection import train_test_split

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5 建模训练与预测
from sklearn.linear_model import LogisticRegression

model = Pipeline([
	('pre', pre),
	('model', LogisticRegression(max_iter=1000, random_state=42))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 6 模型评估
from sklearn.metrics import accuracy_score, precision_score, recall_score

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=0)
rec = recall_score(y_test, y_pred, pos_label=0)
print(f'acc: {acc} prec: {prec} rec: {rec}')

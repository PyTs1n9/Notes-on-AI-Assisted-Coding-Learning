import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()  # 30 个特征 + target（0 = 恶性，1 = 良性）
print(df.shape)  # (569, 31)

# 以下为题目提供的脏数据生成代码，不得修改
rng = np.random.RandomState(42)
n = len(df)
df.loc[rng.rand(n) < 0.04, 'mean texture'] = np.nan
df.loc[rng.choice(n, 8, replace=False), 'mean radius'] = -1.0
df.loc[rng.choice(n, 6, replace=False), 'mean perimeter'] *= 100
df = pd.concat([df, df.sample(12, random_state=0)], ignore_index=True)

import matplotlib.pyplot as plt

print(df['target'].value_counts().sort_index())  # 0: 212, 1: 357
print(df.groupby('target')[['mean radius', 'mean texture', 'mean perimeter']].agg(['mean', 'std']).round(2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for label, name, color in [(0, 'malignant (0)', 'red'), (1, 'benign (1)', 'blue')]:
	axes[0].hist(df.loc[df['target'] == label, 'mean radius'], bins=25, alpha=0.6, label=name, color=color)
axes[0].set_title('mean radius by class')
axes[0].set_xlabel('mean radius')
axes[0].set_ylabel('count')
axes[0].legend()

axes[1].scatter(df['mean radius'], df['mean texture'], c=df['target'], cmap='coolwarm', s=12, alpha=0.7)
axes[1].set_title('mean radius vs mean texture')
axes[1].set_xlabel('mean radius')
axes[1].set_ylabel('mean texture')
plt.tight_layout()
plt.show()

# 脏数据诊断
print('重复行数:', df.duplicated().sum())
print('mean radius 非法值条数:', (df['mean radius'] <= 0).sum())
print('mean texture 缺失数:', df['mean texture'].isnull().sum())
print('mean perimeter 最大值:', df['mean perimeter'].max())

# Step 3 清洗与预处理
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 3.1 删除重复行
df = df.drop_duplicates().reset_index(drop=True)
print(df.shape)

# 3.2 mean radius 非法值置为缺失值
df.loc[df['mean radius'] <= 0, 'mean radius'] = np.nan

# 3.3 mean perimeter 极端值置为缺失：周长 ≈ 2π × 半径，正常上限约 190，> 300 必为量纲错误
PERIM_MAX = 300
df.loc[df['mean perimeter'] > PERIM_MAX, 'mean perimeter'] = np.nan
print('剩余异常值:', (df['mean perimeter'] > PERIM_MAX).sum())

# 3.4 预处理器：中位数填充 + 标准化
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
	('clf', LogisticRegression(max_iter=1000, random_state=42))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 6 模型评估
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=0)
rec = recall_score(y_test, y_pred, pos_label=0)
print(f'Accuracy : {acc:.4f}')
print(f'Precision: {prec:.4f}  (malignant)')
print(f'Recall   : {rec:.4f}  (malignant)')
print(confusion_matrix(y_test, y_pred))

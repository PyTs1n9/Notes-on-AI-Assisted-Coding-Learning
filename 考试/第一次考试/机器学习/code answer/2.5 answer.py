import numpy as np
import pandas as pd
from IPython.core.pylabtools import figsize
from sklearn.datasets import load_diabetes

data = load_diabetes(as_frame=True)
df = data.frame.copy()  # 列名为 age/sex/bmi/bp/s1~s6/target，目标列已命名为 target             # (442, 11)

# 以下为题目提供的脏数据生成代码，不得修改
rng = np.random.RandomState(42)
n = len(df)
sex_map = {v: f'group_{i}' for i, v in enumerate(sorted(df['sex'].unique()))}
df['sex'] = df['sex'].map(sex_map)
for col in ['bmi', 'bp', 's1', 's5']:
	df.loc[rng.rand(n) < 0.05, col] = np.nan
df.loc[rng.choice(n, 8, replace=False), 'bp'] = 0.5
variant = rng.rand(n) < 0.3
df.loc[variant, 'sex'] = df.loc[variant, 'sex'].map(lambda s: ' ' + s.upper())
df = pd.concat([df, df.sample(15, random_state=0)], ignore_index=True)

import matplotlib.pyplot as plt

print(df.describe().T.round(3))

# target 直方图 + bmi-target 散点图
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(df['target'], bins=30, color='#456789', edgecolor='white')
axes[0].set_title('Distribution of target')
axes[0].set_xlabel('target')
axes[0].set_ylabel('count')

axes[1].scatter(df['bmi'], df['target'], s=12, alpha=0.6, color='#456789')
axes[1].set_title('bmi - target')
axes[1].set_xlabel('bmi')
axes[1].set_ylabel('target')

plt.tight_layout()

# 脏数据诊断
print('重复行数:', df.duplicated().sum())              # 15
print('sex 取值:', df['sex'].unique())                 # 4 种写法
print('缺失值:\n', df.isnull().sum())                  # bmi/bp/s1/s5 各约 5%
print('bp 最大值:', df['bp'].max())                    # 0.5，正常范围约 [-0.11, 0.13]
print('bp=0.5 的条数:', (df['bp'] == 0.5).sum())       # 9

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = df.drop_duplicates().reset_index(drop=True)
print('清洗后形状:', df.shape)

df['sex'] = df['sex'].str.strip().str.lower()
print('sex 取值:', df['sex'].unique())

q1, q3 = df['bp'].quantile([0.25, 0.75])
iqr = q3 - q1
upper = q3 + 1.5 * iqr
df.loc[df['bp'] > upper, 'dp'] = np.nan
print('bp 上界:', round(upper, 4), '剩余异常值:', (df['bp'] > upper).sum())
num_cols = [c for c in df.columns if c not in ('sex', 'target')]
sex_cols = ['sex']

num_pipe = Pipeline([
	('imputer', SimpleImputer(strategy='median')),
	('scaler', StandardScaler())
])

sex_pipe = Pipeline([
	('imputer', SimpleImputer(strategy='most_frequent')),
	('onehot', OneHotEncoder(handle_unknown='ignore'))
])

pre = ColumnTransformer([
	('num', num_pipe, num_cols),
	('sex', sex_pipe, sex_cols)
])

from sklearn.model_selection import train_test_split

X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)
print(X_train.shape, X_test.shape)

from sklearn.ensemble import RandomForestRegressor

model = Pipeline([
	('prep', pre),
	('reg', RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MAE: {mae:.2f}')
print(f'R2: {r2:.3f}')

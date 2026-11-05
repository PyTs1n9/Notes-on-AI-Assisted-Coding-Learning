import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

# Step 1：加载数据并合并特征与目标
data = load_diabetes(as_frame=True)
df = data.frame.copy()

# 以下为题目提供的脏数据生成代码，不得修改
rng = np.random.RandomState(42)
n = len(df)
sex_map = {v: f'group_{i}' for i, v in enumerate(sorted(df['sex'].unique()))}
df['sex'] = df['sex'].map(sex_map)  # sex 转为字符串类别
for col in ['bmi', 'bp', 's1', 's5']:
	df.loc[rng.rand(n) < 0.05, col] = np.nan  # 约 5% 缺失值
df.loc[rng.choice(n, 8, replace=False), 'bp'] = 0.5  # 8 个极端值
variant = rng.rand(n) < 0.3
df.loc[variant, 'sex'] = df.loc[variant, 'sex'].map(lambda s: ' ' + s.upper())
df = pd.concat([df, df.sample(15, random_state=0)], ignore_index=True)  # 15 条重复行

# Step 2 EDA 与诊断
print(df.describe().T.round(3))
print(df['target'].value_counts().sort_index())  # 0: 212, 1: 357
print(df.groupby('target')[['mean radius', 'mean texture', 'mean perimeter']].agg(['mean', 'std']).round(2))
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(df['target'], bins=30, color='#456789', edgecolor='white')
axes[0].set_title('target img')
axes[0].set_xlabel('value')
axes[0].set_ylabel('count')

axes[1].scatter(df['bmi'], df['target'], color='#456789', alpha=0.5)
axes[1].set_title('target-bmi')
axes[1].set_xlabel('bmi')
axes[1].set_ylabel('target')

plt.tight_layout()
plt.savefig('2-5 img.png', dpi=150)

# 诊断重复行
print('重复行', df.duplicated().sum())
# 类别写法

# 缺失值统计
print('缺失值', df.isnull().sum())

# bp异常值统计
Q1 = df['bp'].quantile(0.25)
Q3 = df['bp'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
out = df[(df['bp'] < lower) | (df['bp'] > upper)]
print(out)

# Step 3 清洗与预处理
df = df.drop_duplicates().reset_index(drop=True)
df['sex'] = df['sex'].str.strip().str.lower()

num_cols = [c for c in df.columns if c not in ('sex', 'target')]
sex_cols = ['sex']
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

num_pipe = Pipeline([
	('impute', SimpleImputer(strategy='median')),
	('scaler', StandardScaler())
])

sex_pipe = Pipeline([
	('impute', SimpleImputer(strategy='most_frequent')),
	('onehot', OneHotEncoder(handle_unknown='ignore'))
])

pre = ColumnTransformer([
	('num', num_pipe, num_cols),
	('sex', sex_pipe, sex_cols)
])

X = df.drop(columns=['target'])
y = df['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
	X, y, random_state=42, test_size=0.2
)

from sklearn.linear_model import LinearRegression

model = Pipeline([
	('pre', pre),
	('reg', LinearRegression())
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MAE: {mae:.2f} R2: {r2:.4f}')

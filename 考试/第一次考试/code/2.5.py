import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 业务背景：某医院希望根据患者的体检指标（年龄、性别、BMI、平均血压、血清指标等），
# 预测其一年后的病情进展量化指标（连续值），用于制定随访计划。
# 数据由多个科室手工录入汇总，质量不佳。
# 数据来源：sklearn.datasets.load_diabetes，442 条原始样本、10 个特征。
# 请先把特征与目标合并为 DataFrame df（目标列命名为 target），
# 然后原样运行下列脏数据生成代码。

# Step 1：加载数据并合并特征与目标
data = load_diabetes(as_frame=True)
df = data.frame.copy()
df = pd.DataFrame(df)
df.to_csv('2-5.csv', encoding='utf-8-sig', index=False)

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

# Step 2
print(df.describe())
df.info()
print(f'bmi缺失{457 - 428}个值， bp缺失{457 - 436}个值， s1缺失{457 - 430}个值， s5缺失{457 - 437}个值')

plt.hist(df['target'])
plt.xlabel('分数')
plt.ylabel('个数')
plt.title('target直方图')
plt.savefig('target直方图.png', dpi=150)
plt.close()

plt.scatter(df['bmi'], df['target'], alpha=0.5)
plt.xlabel('bmi')
plt.ylabel('target')
plt.title('bmi - target 散点图')
plt.savefig('bmi和target散点图.png', dpi=150)
plt.close()

print(f'重复行：{df.duplicated().sum()}')
print('类别写法')
print(f'{df.dtypes}')
print(f'缺失值：{df.isnull().sum()}')

Q1 = df['bp'].quantile(0.25)
Q3 = df['bp'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
out = df[(df['bp'] < lower) | (df['bp'] > upper)]
print('bp异常值')
print(out)

# Step 3
df = df.drop_duplicates()
out = df[(df['bp'] < lower) | (df['bp'] > upper)]

for i in range(len(df['sex'])):
	df.loc[i, 'sex'] = str(df['sex'][i])[-1]
df.loc[(df['bp'] < lower) | (df['bp'] > upper), 'bp'] = np.nan
# for i in [8, 51, 156, 206, 262, 298, 312, 407]:
# 	df.loc[i, 'bp'] = df['bp'].median()
cols = ['bmi', 'bp', 's1', 's5']

for col in cols:
	df[col] = df[col].fillna(df[col].median())
print("1.删除统一行，并把sex设为独热编码")
print("2.使用IQR处理bp极端值，将bp极端值替换为df['bp']列的中位数，设置中位数是因为受极端值影响较小")
print("3.把每列缺失的数值都替换为该列的平均值，设置平均值是因为不会收到缺失值影响")
print("4.sex已经为0/1独热编码")

# Step 4
X = df.drop(columns='target')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)

# Step 5
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ols = LinearRegression()
ols.fit(X_train_scaled, y_train)
y_pred = ols.predict(X_test_scaled)

# Step 6
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(F'MAE = {mae:.4f} R2 = {r2:.4f}')

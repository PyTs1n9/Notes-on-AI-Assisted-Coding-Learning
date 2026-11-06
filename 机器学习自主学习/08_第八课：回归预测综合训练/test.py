import pandas as pd
import numpy as np
from sklearn.model_selection import (
	train_test_split, GridSearchCV, KFold, cross_val_score
)
from sklearn.linear_model import (
	LinearRegression, Ridge, Lasso
)
from sklearn.metrics import (
	mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
)

# 读数据 + 体检
df = pd.read_csv('score.csv', encoding='utf-8-sig')
print(df.shape)
print(df.head())
print(df.isnull().sum().sum())

# 特征选择
feature_cols = ['学习时长', '手机使用', '睡眠时长', '课外活动']
X = df[feature_cols]
y = df['考试成绩']

# 划分
X_tr, X_te, y_tr, y_te = train_test_split(
	X, y, test_size=0.2, random_state=42
)
print(X_tr.shape)
print(X_te.shape)

# 三模型训练
models = {
	"OLS": LinearRegression(),
	"Ridge": Ridge(alpha=1),
	"Lasso": Lasso(alpha=0.5, max_iter=10000)
}
for name, m in models.items():
	m.fit(X_tr, y_tr)
	print(f'name: {name:<6} coef_: {m.coef_}')

# 指标表
for name, m in models.items():
	pred = m.predict(X_te)
	mae = mean_absolute_error(y_te, pred)
	mse = mean_squared_error(y_te, pred)
	rmse = root_mean_squared_error(y_te, pred)
	r2 = r2_score(y_te, pred)
	print(f'{name:<6} MAE={mae:.2f}  MSE={mse:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}')

# 交叉验证
cv = KFold(n_splits=5, shuffle=True, random_state=42)
for name, m in models.items():
	scores = cross_val_score(m, X, y, cv=cv)
	print(f'{name:<6} 5折R2分数：{np.round(scores, 4)}  平均={scores.mean():.4f}')

# 网格搜索
gs = GridSearchCV(Ridge(), {'alpha': [0.01, 0.1, 1, 10, 100]}, cv=cv)
gs.fit(X, y)
print('best_params_:', gs.best_params_)
print('best_score_:', gs.best_score_)
for a, s in zip(gs.cv_results_['param_alpha'], gs.cv_results_['mean_test_score']):
	print(f'alpha: {a:<6} 平均R2: {s:.8f}')

# 三句结果分析
print("① 指标表上谁最好、差多少")
print("指标表上 OLS 最好：R2=0.7016，比 Ridge 高 0.0002、比 Lasso 高 0.0067，差距不到 0.01——三个模型基本打平")

print("② alpha 曲线怎么读")
print('找最高行：alpha=0.01的R2最大；看平台：基本都差不多；看尾部：10略降，100直接掉一截，说明100的时候惩罚过头了，需要排掉')

print("③ 必须点到 Lasso 对「课外活动」系数做了什么（看 `coef_`）")
print('Lasso把课外活动的系数压成了-0')

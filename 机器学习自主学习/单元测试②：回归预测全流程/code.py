import numpy as np
import pandas as pd
from sklearn.model_selection import (
	train_test_split, KFold, cross_val_score, GridSearchCV
)
from sklearn.linear_model import (
	LinearRegression, Ridge, Lasso
)
from sklearn.metrics import (
	mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
)

df = pd.read_csv('score.csv', encoding='utf-8-sig')
print(df.head())
print(df.shape)
print(df.isnull().sum().sum())

X = df[['学习时长', '手机使用', '睡眠时长', '课外活动']]
y = df['考试成绩']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
print(X_tr.shape, X_te.shape)
print(y_tr.shape, y_te.shape)

models = {
	"OLS": LinearRegression(),
	"Ridge": Ridge(alpha=1),
	"Lasso": Lasso(alpha=0.5, max_iter=10000)
}

for name, m in models.items():
	m.fit(X_tr, y_tr)
	pred = m.predict(X_te)
	mae = mean_absolute_error(y_te, pred)
	mse = mean_squared_error(y_te, pred)
	rmse = root_mean_squared_error(y_te, pred)
	r2 = r2_score(y_te, pred)
	print(f"Name: {name:<6}coef_: {m.coef_} ")
	print(f"Name: {name:<6}MAE: {round(mae, 2)} MSE: {round(mse, 2)} RMSE: {round(rmse, 2)} R2: {round(r2, 4)}")
print("一句话点评：Lasso直接把课外活动的系数压成0")


df = pd.read_csv('sales.csv', encoding='utf-8-sig')
X = df[['广告投入', '优惠券', '店员人数', '会员数', '客流量']]
y = df['销量']
cv = KFold(n_splits=5, shuffle=True, random_state=7)

models = {
	"OLS": LinearRegression(),
	"Ridge": Ridge(alpha=1),
	"Lasso": Lasso(alpha=1, max_iter=10000)
}

for name, m in models.items():
	scores = cross_val_score(m, X, y, cv=cv)
	print(f"Name: {name:<6} Scores: {np.round(scores, 4)} 平均分: {round(scores.mean(), 4)}")

gs = GridSearchCV(Ridge(), {'alpha': [0.01, 0.1, 1, 10, 100]}, cv=cv)
gs.fit(X, y)
print(gs.best_params_)
print(gs.best_score_)
for a,s in zip(gs.cv_results_['param_alpha'],gs.cv_results_['mean_test_score']):
	print(f'Alpha: {a:<6} Scores: {s}')
print('0.01,0.1,1.0,10基本平坦。α等于100的时候直接掉。说明100是坏区间')
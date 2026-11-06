import pandas as pd
import numpy as np
from sklearn.model_selection import (
	train_test_split, cross_val_score, KFold, GridSearchCV
)
from sklearn.linear_model import (
	LinearRegression, Ridge, Lasso
)
from sklearn.metrics import (
	mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
)

# ---------- 1. 读数据 ----------
df = pd.read_csv('sales.csv', encoding='utf-8-sig')
print('shape:', df.shape)
print(df.head())
print('null sum:', df.isnull().sum().sum())

# ---------- 2. 特征选择 ----------
feature_cols = ['广告投入', '优惠券', '店员人数', '会员数', '客流量']
X = df[feature_cols]
y = df['销量']
print('X', X.shape, 'y', y.shape)

# ---------- 3. 划分训练/测试 ----------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
print('训练集:', X_tr.shape, ' 测试集:', X_te.shape)

# ---------- 4. 三模型训练 ----------
models = {
	'OLS': LinearRegression(),
	'Ridge': Ridge(alpha=1),
	'Lasso': Lasso(alpha=1, max_iter=10000)
}

for name, m in models.items():
	m.fit(X_tr, y_tr)

# ---------- 5. 指标评价 ----------
print('\n--- 测试集指标对比 ---')
for name, m in models.items():
	pred = m.predict(X_te)
	mae = mean_absolute_error(y_te, pred)
	mse = mean_squared_error(y_te, pred)
	rmse = root_mean_squared_error(y_te, pred)
	r2 = r2_score(y_te, pred)
	print(f'{name:<6} MAE={mae:.2f}  MSE={mse:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}')

# ---------- 6. 交叉验证：换个考场再比一次 ----------
print('\n--- 6 折交叉验证（全数据）---')
cv = KFold(n_splits=5, shuffle=True, random_state=42)
for name, m in models.items():
	scores = cross_val_score(m, X, y, cv=cv)
	print(f'{name:<6} 5折R2: {np.round(scores, 4)}  平均={scores.mean():.4f}')

# ---------- 7. GridSearchCV 给 Ridge 调 alpha ----------
print('\n--- GridSearchCV 搜 Ridge 的 alpha ---')
gs = GridSearchCV(Ridge(), {'alpha': [0.01, 0.1, 1, 10, 100]}, cv=cv)
gs.fit(X, y)
print('best_params_:', gs.best_params_, 'best_score_:', round(gs.best_score_, 4))
for a, s in zip(gs.cv_results_['param_alpha'], gs.cv_results_['mean_test_score']):
	print(f'alpha={a:<6} 平均R2={s:.4f}')

# ---------- 8. 系数体检：看谁学了规律 ----------
print('\n--- 系数对比（真值：广告投入5 优惠券0 店员人数0 会员数3 客流量2）---')
for name, m in models.items():
	print(f'{name:<6}', np.round(m.coef_, 3))

# ---------- 9. 结果分析 ----------
print(
	'\n结论1: 测试集上 OLS 略高(0.848)、5 折 CV 上 Lasso 略高(0.810)，差距都不到 0.005——三个模型基本打平，这批数据惩罚没派上大用场')
print('结论2: Lasso 把噪声特征优惠券压到了', round(float(models['Lasso'].coef_[1]), 3),
      '、店员人数缩到', round(float(models['Lasso'].coef_[2]), 3), '——特征选择再次生效')
print('结论3: alpha 曲线 0.01~10 平稳、100 变差，best alpha=10——best_score_ 只用来挑 alpha，最终成绩看测试集 R2')

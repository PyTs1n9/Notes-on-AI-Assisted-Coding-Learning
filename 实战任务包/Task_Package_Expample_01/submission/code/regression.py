import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from sklearn.model_selection import (
	train_test_split, cross_val_score, KFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
	LinearRegression, Lasso, Ridge
)
from sklearn.metrics import (
	mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
)

plt.rcParams['font.sans-serif'] = ['SimHei']

df = pd.read_csv('../data_outputs/hour_clean.csv', encoding='utf-8-sig')

drop_list = ["cnt", "casual", "registered", "instant"]
X = df.drop(columns=drop_list)
y = df["cnt"]

X['dteday'] = pd.to_datetime(X['dteday'], errors='coerce')
X['date_year'] = X['dteday'].dt.year
X['date_month'] = X['dteday'].dt.month
X['date_day'] = X['dteday'].dt.day

X = X.drop('dteday', axis=1)
X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# models = {
# 	'ols': LinearRegression(),
# 	'Ridge': Ridge(alpha=1.0),
# 	'Lasso': Lasso(alpha=1.0, max_iter=10000)
# }
#
# rows_metrics = []
# pred_df = pd.DataFrame({'y_true': y_test.values})
# for name, m in models.items():
# 	m.fit(X_train_scaled, y_train)
# 	pred = m.predict(X_test_scaled)
# 	mae = mean_absolute_error(y_test, pred)
# 	mse = mean_squared_error(y_test, pred)
# 	rmse = root_mean_squared_error(y_test, pred)
# 	r2 = r2_score(y_test, pred)
# 	rows_metrics.append({
# 		'Model': name,
# 		'MAE': mae,
# 		'MSE': mse,
# 		'RMSE': rmse,
# 		'R2': r2
# 	})
# 	pred_df[f'{name}_pred'] = pred
#
# regression_metrics = pd.DataFrame(rows_metrics)
# regression_metrics.to_csv('../metrics/regression_metrics.csv', encoding='utf-8-sig', index=False)
# regression_predictions = pd.DataFrame(pred_df)
# regression_predictions.to_csv('../data_outputs/regression_predictions.csv', encoding='utf-8-sig', index=False)
#
# models = {'Ridge': Ridge(), 'Lasso': Lasso()}
# alpha_list = [0.01, 0.1, 1, 10, 100]
# cv = KFold(n_splits=5, shuffle=True, random_state=42)
# for a in alpha_list:
# 	ridge_score = cross_val_score(Ridge(alpha=a), X_train_scaled, y_train, cv=cv)
# 	print(f'Ridge alpha={a:<6} 平均CV分数: {ridge_score.mean():.4f}')
# for a in alpha_list:
# 	lasso_score = cross_val_score(Lasso(alpha=a), X_train_scaled, y_train, cv=cv)
# 	print(f'Lasso alpha={a:<6} 平均CV分数: {lasso_score.mean():.4f}')
#
# gs = GridSearchCV(Ridge(), {'alpha': alpha_list}, cv=cv)
# gs.fit(X_train_scaled, y_train)
# print('Ridge最优alpha参数：', gs.best_params_)
# print('Ridge平均CV分数：', gs.best_score_)
# gs = GridSearchCV(Lasso(), {'alpha': alpha_list}, cv=cv)
# gs.fit(X_train_scaled, y_train)
# print('Lasso最优alpha参数：', gs.best_params_)
# print('Lasso平均CV分数：', gs.best_score_)

m = LinearRegression()
m.fit(X_train_scaled, y_train)
pred = m.predict(X_test_scaled)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, s=5, alpha=0.5)
lims = [0, max(y_test.max(), pred.max()) * 1.02]
plt.plot(lims, lims, 'r--', lw=1, label='y = x')
plt.xlabel('y_test')
plt.ylabel('pred')
plt.title('y_test - pred')
plt.legend()
plt.tight_layout()
plt.savefig('../figures/regression_scatter.png', dpi=150)
plt.close()

metrics = pd.read_csv('../metrics/regression_metrics.csv', encoding='utf-8-sig')
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(metrics['Model'], metrics['MAE'])
ax.set_ylabel('MAE')
ax.set_title('Model - MAE')
for i, v in enumerate(metrics['MAE']):
	ax.text(i, v, f'{v:.1f}')
plt.tight_layout()
plt.savefig('../figures/regression_model_compare.png', dpi=150)
plt.close()

res = pd.DataFrame({'y_true': y_test.values, 'pred': pred})
res['abs_err'] = (res['y_true'] - res['pred']).abs()
res.index = y_test.index
top2 = res.nlargest(2, 'abs_err')
print(top2)
print(df.loc[top2.index, ['dteday', 'hr', 'weathersit', 'temp', 'hum', 'cnt']])
# 时段与需求是非线性关系、缺少促销/活动等高影响变量、极端峰值本就是线性外推最难的部分。

# 2.6 三个模型几乎打平，3个模型分数指标都相似，产生的偏差原因为cnt与天气，时段多为非线性关系，且缺少促销，活动等变量，改进方向为非线性模型或时段交互特征
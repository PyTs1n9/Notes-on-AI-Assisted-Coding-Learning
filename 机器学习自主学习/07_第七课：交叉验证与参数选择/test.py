from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np

X, y = make_regression(n_samples=80, n_features=5, n_informative=2, noise=30, random_state=7)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores_l5 = cross_val_score(LinearRegression(), X, y, cv=cv)
print(np.round(scores_l5, 4))

manual = []
for train_idx, test_idx in cv.split(X):
	m = LinearRegression().fit(X[train_idx], y[train_idx])
	manual.append(m.score(X[test_idx], y[test_idx]))
manual = np.array(manual)
print('手动循环:', np.round(manual, 4))
print('逐折一致:', np.allclose(manual, scores_l5))

scores_r5 = cross_val_score(Ridge(alpha=1), X, y, cv=cv)
print(np.round(scores_r5, 4))
print('LR 平均分:', round(scores_l5.mean(), 4))
print('Ridge 平均分:', round(scores_r5.mean(), 4))
print('结论：Ridge 平均分比 LR 高', round(scores_r5.mean() - scores_l5.mean(), 4))

gs = GridSearchCV(Ridge(), {'alpha': [0.01, 0.1, 1, 10, 100]}, cv=cv)
gs.fit(X, y)
print(gs.best_params_)
print(gs.best_score_)
for a, s in zip(gs.cv_results_['param_alpha'], gs.cv_results_['mean_test_score']):
	print(f"alpha:{a:<6} R2:{s:.4f}")

neg = cross_val_score(LinearRegression(), X, y, cv=cv, scoring='neg_mean_squared_error')
print(neg)
print('因为使用的指标器是neg_mean_squared_error，sklearn中，分数越大越好，所以返回 −MSE，取负号还原')
print('手算均值 = 5 折之和/5 =', round(float(np.round(scores_l5, 4).sum()) / 5, 4))
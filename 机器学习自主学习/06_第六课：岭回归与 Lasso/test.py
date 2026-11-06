import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso

x = np.array([1, 2, 3, 4])
y = 3 * x + 2
X = x.reshape(-1, 1)
model = Ridge(alpha=10)
model.fit(X, y)
pred = model.predict([[3]])
print(model.coef_)  # 15/(5+α)=1
print(model.intercept_)  # b=2*x平均+2=5+2=7
print(pred)  # 1*3+7=10

rng = np.random.default_rng(7)
X4 = rng.normal(size=(20, 4))  # 4 个特征，x3 x4 是纯噪声
y4 = 4 * X4[:, 0] + 1.5 * X4[:, 1] + rng.normal(scale=0.5, size=20)  # 真实规律 4x1 + 1.5x2

model_lr = LinearRegression()
model_ri = Ridge(alpha=1)
model_la = Lasso(alpha=0.3, max_iter=10000)
model_lr.fit(X4, y4)
model_ri.fit(X4, y4)
model_la.fit(X4, y4)

tab = pd.DataFrame({
	'真实系数': [4, 1.5, 0, 0],
	'OLS': np.round(model_lr.coef_, 3),
	'Ridge': np.round(model_ri.coef_, 3),
	'Lasso': np.round(model_la.coef_, 3)
}, index=['x1', 'x2', 'x3', 'x4'])
print(tab)
print("ols,ridge被噪声骗了，lasso把噪声压成了0，小")

print((model_la.coef_[2] == 0)&(model_la.coef_[3] == 0))
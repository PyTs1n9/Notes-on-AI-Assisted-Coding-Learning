import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
	mean_absolute_error,
	mean_squared_error,
	root_mean_squared_error,
	r2_score
)
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
x_raw = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([11, 14, 19, 26, 29, 33, 39, 43, 48, 53])
X = x_raw.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.3, random_state=0
)
print(X_train.shape)
print(X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))
print("MSE:", mean_squared_error(y_test, pred))
print("RMSE:", root_mean_squared_error(y_test, pred))
print("R2:", r2_score(y_test, pred))
print("score", model.score(X_test, y_test))

tab = pd.DataFrame({
	"真实值": y_test,
	"预测值": np.round(pred, 2),
	"误差": np.round(pred - y_test, 2)
})
# 手算 MAE = (0.88 + 0.05 + 0.27) ÷ 3 = 0.40
# 实测 MAE = 0.4035874439461864 → 保留 2 位 = 0.40
print(tab.to_string(index=False))
plt.plot(y_test, 'o-', label='真实值')
plt.plot(pred, 'o--', label='预测值')
plt.xlabel('广告投入')
plt.ylabel('商品销量')
plt.title('广告投入 - 商品销量')
plt.legend()
plt.tight_layout()
plt.savefig('广告对比.png')

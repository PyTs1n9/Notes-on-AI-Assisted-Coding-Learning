import numpy as np
from sklearn.linear_model import LinearRegression

x = [2, 4, 6, 8]  # 广告投入（万元）
y = [15, 25, 35, 45]  # 销量（百件）

x_raw = np.array(x)
y = np.array(y)

X = x_raw.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)

print("w:", model.coef_)
print("b:", model.intercept_)

x_new = np.array([[7]])
y_pred = model.predict(x_new)
print("广告投入7万元时候，销量有", y_pred[0], "百件")

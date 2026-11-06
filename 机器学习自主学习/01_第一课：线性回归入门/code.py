import numpy as np
from sklearn.linear_model import LinearRegression

x_raw = np.array([1, 2, 3, 4, 5])
y = np.array([10, 13, 16, 19, 22])

X = x_raw.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)

print("斜率：", model.coef_)
print("截距：", model.intercept_)

x_new = np.array([[6], [7]])
y_pred = model.predict(x_new)

print("学习6小时", y_pred[0], "分")
print("学习7小时", y_pred[1], "分")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x_raw = np.arange(1, 21)
noise = np.array([0.5, -0.7, 0.2, 0.9, -0.4,
                  0.6, -0.3, 0.8, -0.9, 0.1,
                  0.7, -0.5, 0.3, -0.8, 0.4,
                  0.9, -0.2, 0.6, -0.6, 0.2])
y = 3 * x_raw + 7 + noise
X = x_raw.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)
print("X_train形状：", X_train.shape, "y_train长度", y_train.shape)
print("X_test形状：", X_test.shape, "y_test长度", y_test.shape)

idx = np.arange(20)
idx_train, idx_test = train_test_split(
	idx, test_size=0.2, random_state=42
)
print("测试集行号：", sorted(idx_test.tolist()))
print("两边是否重叠：", len(set(idx_train) & set(idx_test)) != 0)
print("两边合起来完整吗:", sorted(np.concatenate([idx_train, idx_test]).tolist()) == list(range(20)))

model = LinearRegression()
model.fit(X_train, y_train)
print("斜率 w = ", model.coef_, " 截距 b = ", round(model.intercept_, 2))

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("训练集 R² =", round(train_score, 4))
print("测试集 R² =", round(test_score, 4))
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)
noise = np.array(rng.uniform(-1, 1, 30).round(1))
x = np.arange(1, 31, dtype=float)
y = 5 * x + 20 + noise
X = x.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.3, random_state=42
)
print("X_train形状：", X_train.shape)
print("X_test形状：", X_test.shape)

idx = np.arange(30)
idx_train, idx_test = train_test_split(
	idx, test_size=0.3, random_state=42
)
print("测试集行号:", sorted(idx_test.tolist()))
print("是否重复:", len(set(idx_train) & set(idx_test)) != 0)
print("是否完整:", sorted(np.concatenate([idx_train, idx_test]).tolist()) == list(range(30)))

model = LinearRegression()
model.fit(X_train, y_train)
print("w = ", model.coef_)
print("b = ", round(model.intercept_, 2))

print("训练集r2:", model.score(X_train, y_train))
print("训练集r2:", model.score(X_test, y_test))

print("为什么训练集 R² 和测试集 R² 不会完全一样？")
print("因为训练集和测试集的样本是不一样的")

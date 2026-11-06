import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = 5 * X.ravel() + 20

X_train, X_test, y_train, y_test = train_test_split(
	X, y, random_state=0, test_size=0.3
)
print(X_train.shape)
print(X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

print("coef_ ", model.coef_)
print("intercept_ ", model.intercept_)

X_new = [[3]]
pred = model.predict(X_new)
print(pred)

print(model.score(X_train,y_train))
print(model.score(X_test,y_test))

# np.concatenate 这一块不会了

X_all = np.concatenate((X_train, X_test))   # 函数①：双层括号，把两段 X 首尾拼回
y_all = np.concatenate((y_train, y_test))   # 两段 y 也拼回
print('X_all.shape =', X_all.shape)          # 核对 1：样本数 = 7 + 3 = 10，不缺
print('排序后 =', np.sort(X_all.ravel()))    # 核对 2：排序后正好 1~10，不重不漏
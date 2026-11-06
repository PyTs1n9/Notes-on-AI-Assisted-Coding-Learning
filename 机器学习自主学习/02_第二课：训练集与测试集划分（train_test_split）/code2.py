import numpy as np                                     # 数组库
from sklearn.model_selection import train_test_split   # 划分函数

X = np.arange(1, 21, dtype=float).reshape(-1, 1)       # 20 个样本的特征
y = 3 * X.ravel() + 7                                  # 标签（ravel 把二维压回一维）

X_a, _, y_a, _ = train_test_split(X, y, test_size=0.2, random_state=42)  # 函数①：种子 42，第 1 次
X_b, _, y_b, _ = train_test_split(X, y, test_size=0.2, random_state=42)  # 函数①：种子 42，第 2 次
X_c, _, y_c, _ = train_test_split(X, y, test_size=0.2, random_state=7)   # 函数①：换种子 7

print("同种子(42)两次，划分一样吗:", np.array_equal(X_a, X_b))   # array_equal 逐元素比较两个数组
print("换种子(7)之后，划分一样吗:", np.array_equal(X_a, X_c))
print("种子 42 的训练集:", sorted(X_a.ravel().astype(int).tolist()))  # 打印排序后的训练集 x 值（转纯数字）
print("种子 7  的训练集:", sorted(X_c.ravel().astype(int).tolist()))
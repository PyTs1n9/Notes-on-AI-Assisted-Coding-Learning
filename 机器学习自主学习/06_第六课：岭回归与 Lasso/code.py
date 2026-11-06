# ============ 第六课：岭回归与 Lasso ============
import numpy as np                         # 数组工具
import pandas as pd                        # 表格工具（摆系数对比表）
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # 三个模型

# ---------- 0. 种子小实验：同种子→同结果，不同种子→不同结果 ----------
print('===== 种子小实验 =====')
a = np.random.default_rng(42).normal(size=3)   # 种子 42，第一次
b = np.random.default_rng(42).normal(size=3)   # 种子 42，第二次
c = np.random.default_rng(7).normal(size=3)    # 种子 7
print('种子 42 第一次 :', np.round(a, 4).tolist())  # tolist()：numpy 数打印得干净
print('种子 42 第二次 :', np.round(b, 4).tolist())  # 和上一行一模一样
print('种子 7        :', np.round(c, 4).tolist())  # 换种子就换数

# ---------- 第一部分：手算对照，看 alpha 怎么压系数 ----------
print()
print('===== 第一部分：y = 2x + 1，Ridge 把系数往 0 压 =====')
x = np.array([1, 2, 3, 4])                 # 特征
y = 2 * x + 1                              # 标签，严格 y = 2x + 1，没噪声
X = x.reshape(-1, 1)                       # 转二维（第一课函数②）

ols = LinearRegression()                   # 造普通线性回归
ols.fit(X, y)                              # 函数：训练
print('OLS          coef =', ols.coef_.tolist(),              # coef_：系数（属性，不加括号）
      'intercept =', round(float(ols.intercept_), 4))        # intercept_：截距

for alpha in [0.5, 1, 5, 10]:              # 惩罚力度从轻到重
    ridge = Ridge(alpha=alpha)             # 函数①：造岭回归模型
    ridge.fit(X, y)                        # 训练
    print(f'Ridge α={alpha:<4} coef =', np.round(ridge.coef_, 4).tolist(),
          'intercept =', round(float(ridge.intercept_), 4),
          '| 手算 10/(5+α) =', round(10 / (5 + alpha), 4))    # 手算公式对照

ridge5 = Ridge(alpha=5).fit(X, y)          # 造模型+训练一步完成
print('Ridge(α=5) 预测 x=3 :', np.round(ridge5.predict([[3]]), 4).tolist(),  # 函数③
      '| 手算 1.0×3 + 3.5 = 6.5')

# ---------- 第二部分：4 个特征里 2 个是纯噪声，三模型对比 ----------
print()
print('===== 第二部分：噪声特征对比 =====')
rng = np.random.default_rng(42)            # 种子固定，咱俩结果一样
X4 = rng.normal(size=(20, 4))              # 20 个样本 × 4 个特征（x1 x2 有用，x3 x4 噪声）
y4 = 3 * X4[:, 0] - 2 * X4[:, 1] + rng.normal(scale=0.5, size=20)  # 真实规律：3x1 − 2x2 + 小噪声

ols4 = LinearRegression().fit(X4, y4)      # 普通线性回归
ridge4 = Ridge(alpha=1).fit(X4, y4)        # 函数①：岭回归
lasso4 = Lasso(alpha=0.1, max_iter=10000).fit(X4, y4)  # 函数②：Lasso 回归

tab = pd.DataFrame({                       # 系数对比表：每行一个特征
    '真实系数': [3, -2, 0, 0],             # 生成数据时的真值（x3 x4 是噪声，真系数 0）
    'OLS':   np.round(ols4.coef_, 3),      # 普通线性回归的估计
    'Ridge': np.round(ridge4.coef_, 3),    # 岭回归的估计
    'Lasso': np.round(lasso4.coef_, 3),    # Lasso 的估计
}, index=['x1', 'x2', 'x3(噪声)', 'x4(噪声)'])
print(tab.to_string())                     # 打印对比表

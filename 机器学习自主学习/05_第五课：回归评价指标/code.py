# ============ 第五课：回归评价指标 MAE / MSE / RMSE / R² ============
import numpy as np                        # 数组工具
import pandas as pd                       # 表格工具（只用来把结果摆成对照表）
from sklearn.model_selection import train_test_split   # 划分训练集/测试集
from sklearn.linear_model import LinearRegression      # 线性回归模型
from sklearn.metrics import (                          # 四个评价指标函数
    mean_absolute_error,        # 函数①：MAE
    mean_squared_error,         # 函数②：MSE
    root_mean_squared_error,    # 函数③：RMSE
    r2_score,                   # 函数④：R²
)
import matplotlib
matplotlib.use('Agg')                     # 无界面环境也能画图（只存文件不弹窗）
import matplotlib.pyplot as plt           # 画图工具
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 中文字体，防止中文变方框
plt.rcParams['axes.unicode_minus'] = False                     # 负号正常显示

# ---------- 第一部分：小例子，手算对照 ----------
print('===== 第一部分：3 个样本的手算对照 =====')
y_true = np.array([10, 20, 30])           # 真实值
y_pred = np.array([12, 18, 33])           # 预测值（故意猜得有偏差）

print('MAE  =', mean_absolute_error(y_true, y_pred))    # 函数①
print('MSE  =', mean_squared_error(y_true, y_pred))     # 函数②
print('RMSE =', root_mean_squared_error(y_true, y_pred))# 函数③
print('R2   =', r2_score(y_true, y_pred))               # 函数④

# ---------- 第二部分：接回线性回归全流程 ----------
print()
print('===== 第二部分：学时 → 成绩，测试集上打分 =====')
x_raw = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])       # 学习时长
y = np.array([11, 12, 16, 21, 20, 26, 28, 30, 36, 36])  # 成绩（在 y=3x+7 上下波动）
X = x_raw.reshape(-1, 1)                                # 转成二维

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0                 # 7 训练 / 3 测试
)

model = LinearRegression()              # 造模型
model.fit(X_train, y_train)             # 只用训练集学习

y_pred_test = model.predict(X_test)     # 对测试集 3 个样本做预测

# 把测试集的真实值 / 预测值 / 误差摆成一张表（误差 = 预测值 - 真实值）
tab = pd.DataFrame({
    '真实值': y_test,
    '预测值': np.round(y_pred_test, 2),   # 保留 2 位小数，方便手算对照
    '误差': np.round(y_pred_test - y_test, 2),
})
print(tab.to_string(index=False))

print()
print('--- 测试集指标 ---')
print('MAE  =', mean_absolute_error(y_test, y_pred_test))    # 函数①
print('MSE  =', mean_squared_error(y_test, y_pred_test))     # 函数②
print('RMSE =', root_mean_squared_error(y_test, y_pred_test))# 函数③
print('R2   =', r2_score(y_test, y_pred_test))               # 函数④

print()
print('--- 训练集指标（对照用） ---')
y_pred_train = model.predict(X_train)                       # 对训练集也预测一遍
print('训练集 MAE =', mean_absolute_error(y_train, y_pred_train))
print('训练集 R2  =', r2_score(y_train, y_pred_train))
print('model.score(X_test, y_test) =', model.score(X_test, y_test))  # 函数⑤：回归下就是 R²

# ---------- 第三部分：真实值 vs 预测值 对比图 ----------
plt.figure(figsize=(6, 4))                       # 画布大小
plt.plot(y_test, 'o-', label='真实值')           # 圆点+实线画真实值
plt.plot(y_pred_test, 's--', label='预测值')     # 方块+虚线画预测值
plt.title('测试集：真实值 vs 预测值')            # 图标题
plt.xlabel('测试样本序号')                       # 横轴名
plt.ylabel('成绩')                               # 纵轴名
plt.legend()                                     # 显示图例
plt.tight_layout()                               # 自动收紧边距
plt.savefig('真实vs预测.png', dpi=120)           # 存成图片文件
print()
print('图片已保存：真实vs预测.png')

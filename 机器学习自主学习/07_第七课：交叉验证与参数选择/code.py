# ============ 第七课：交叉验证与参数选择 ============
import numpy as np                                            # 数组工具
from sklearn.datasets import make_regression                  # 造回归数据（第六课见过）
from sklearn.linear_model import LinearRegression, Ridge      # 两个模型
from sklearn.model_selection import (                         # 本课三个新工具都从这里 import
    train_test_split, cross_val_score, KFold, GridSearchCV)

# ---------- 造数据：5 个特征，只有 3 个真正有用，2 个是噪声特征 ----------
X, y = make_regression(n_samples=120, n_features=5, n_informative=3, noise=25, random_state=42)
# n_informative=3：120 行样本、5 个特征，其中只有 3 个特征真正参与生成 y，
# 另 2 个是纯噪声特征（呼应第六课的 x3、x4）；noise=25：标签里掺随机噪声
print('X:', X.shape, ' y:', y.shape)                          # (120, 5) (120,)

# ---------- 示例1：换一次划分，分数就变 ----------
print('\n===== 示例1：换一次划分，分数就变 =====')
for seed in [0, 1, 2, 3, 4]:                                  # 5 个种子 = 5 套不同考卷
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)  # 函数①（第一、二课学过）
    m = LinearRegression().fit(X_tr, y_tr)                    # 函数② 每次都重新训练（其实模型一样，划分不同）
    print(f'random_state={seed}  测试集 R2={m.score(X_te, y_te):.4f}')  # 函数③ 同模型不同考卷，分数抖动

# ---------- 示例2：cross_val_score 五折交叉验证 ----------
print('\n===== 示例2：cross_val_score 五折交叉验证 =====')
cv = KFold(n_splits=5, shuffle=True, random_state=42)         # 函数④ 定切法：5 折、洗牌、种子固定可复现
scores5 = cross_val_score(LinearRegression(), X, y, cv=cv)    # 函数⑤ 自动训 5 次、打 5 次分
print('5 折分数:', np.round(scores5, 4))                       # 返回数组，每折一个分数
print('平均分:', round(scores5.mean(), 4))                     # 最终成绩 = 5 个分数取平均

# 手动 KFold 循环对照：证明 cross_val_score 干的就是这件事
manual = []                                                   # 收集手动循环算的分数
for tr_idx, te_idx in cv.split(X):                            # 函数⑥ 每轮吐出 (训练下标, 测试下标)
    m = LinearRegression().fit(X[tr_idx], y[tr_idx])           # 用下标取行，训练
    manual.append(m.score(X[te_idx], y[te_idx]))               # 用下标取行，打分
manual = np.array(manual)                                     # 列表转 numpy 数组，好和 scores5 对比
print('手动循环:', np.round(manual, 4))
print('逐折一致:', np.allclose(manual, scores5))               # np.allclose：两个数组是否几乎相等

# 默认 cv=5 = 不洗牌的 KFold（验证坑①）
d1 = cross_val_score(LinearRegression(), X, y, cv=5)          # 不传切分器，走默认
d2 = []                                                       # 手动用"不洗牌 KFold"复现一遍
for tr_idx, te_idx in KFold(n_splits=5).split(X):
    m = LinearRegression().fit(X[tr_idx], y[tr_idx])
    d2.append(m.score(X[te_idx], y[te_idx]))
print('默认 cv=5 就是不洗牌 KFold:', np.allclose(d2, d1))       # True，默认切法就是这样

# scoring 的符号坑（验证坑②）
neg = cross_val_score(LinearRegression(), X, y, cv=cv, scoring='neg_mean_squared_error')
print('neg_mean_squared_error:', np.round(neg, 2))             # 全是负数！
print('取负号后 = MSE:', np.round(-neg, 2))                    # 取负号才还原成真正的 MSE

# ---------- 示例3：GridSearchCV 搜 alpha ----------
print('\n===== 示例3：GridSearchCV 搜 alpha =====')
gs = GridSearchCV(Ridge(), {'alpha': [0.01, 0.1, 1, 10, 100]}, cv=cv)  # 函数⑦ 5 个候选 × 5 折 = 训 25 次
gs.fit(X, y)                                                  # 函数⑧ 一次 fit 跑完整个搜索
print('best_params_:', gs.best_params_)                       # 最优超参数
print('best_score_:', round(gs.best_score_, 4))               # 最优候选的 CV 平均分
for a, s in zip(gs.cv_results_['param_alpha'], gs.cv_results_['mean_test_score']):
    print(f'alpha={a:<6} 平均 R2={s:.4f}')                     # 每个候选的成绩一览
ols_mean = cross_val_score(LinearRegression(), X, y, cv=cv).mean()
print('同样 CV 下 LinearRegression 平均 R2:', round(ols_mean, 4))  # 和最优 Ridge 对比
print('best_estimator_ 系数:', np.round(gs.best_estimator_.coef_, 4))  # refit 重训好的模型直接可用
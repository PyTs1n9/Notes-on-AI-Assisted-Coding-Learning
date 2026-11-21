# 逻辑回归分类器

[返回示例总目录](../README.md)。每段独立运行，下面是 `BaiQing3.10AI` 的真实输出。

## 函数卡

- **作用**：由当前特征估算各类别的概率并作分类，不是直接预测连续数值。
- **语法**：`LogisticRegression(penalty="l2", *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver="lbfgs", max_iter=100, multi_class="deprecated", verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)`（本机 scikit-learn 1.7.2，`multi_class` 已弃用，不需要传）；`model.fit(X, y, sample_weight=None)`；`model.predict(X)`；`model.predict_proba(X)`。
- **关键参数**：`C` 越小正则化通常越强；`max_iter` 是求解迭代上限；`class_weight="balanced"` 按类别频数反比加权。
- **返回值**：拟合返回模型自身；`predict` 返回类别数组；`predict_proba` 返回每行各类别概率，列顺序为 `classes_`。
- **常见坑**：模型评估用未参与拟合的样本；根据真实标签定位概率列；不能事后用测试结果挑阈值。
- **适用场景**：二分类、多分类、不平衡风险识别。

## 示例 1：订单是否需人工复核

在训练集上拟合预处理和分类器，在未参与训练的订单上看类别和准确率。

运行 [示例01.py](示例01.py)。

```text
复核类别 [0, 1, 0, 1]
测试准确率 1.0
```

## 示例 2：比较两个预先给定的预警阈值

同一个模型，不同阈值会改变报警的数量；这里只观察差异，不用测试数据选阈值。

运行 [示例02.py](示例02.py)。

```text
阈值、漏报、误报 0.35 0 1
阈值、漏报、误报 0.65 1 0
```

## 示例 3：多分类概率按类名取列

将三类请求分给对应窗口，同时正确读出「缴费」类的概率，而不是固定列下标。

运行 [示例03.py](示例03.py)。

```text
概率列顺序 ['咨询', '报修', '缴费']
自动分配 ['缴费', '咨询', '报修']
各条缴费概率 [0.9631, 0.0308, 0.0107]
```

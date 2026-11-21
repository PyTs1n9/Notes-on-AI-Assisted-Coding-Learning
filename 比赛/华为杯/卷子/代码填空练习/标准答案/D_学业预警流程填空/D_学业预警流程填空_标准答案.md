# D 卷 · 学业预警流程填空：标准答案与解析

> 本页是本次练习卷的参考实现，不是官方答案，也不代表已独立掌握。建议先填完题再打开。

[返回试卷](../../D_学业预警流程填空/试卷.md) · [答案索引](../README.md)。下面的实跑输出来自每题参考脚本，使用 `BaiQing3.10AI`；不把参考指标写死为评分线。
每行答案替换对应占位符，其他脚手架不用动。完整的逐题脚本链接可独立运行，产物写入 `参考实现/outputs/`。

## D01 读取、探索与成绩段分布

[完整参考实现](参考实现/D01.py)

| 空位 | 填入 | 为什么 |
| --- | --- | --- |
| `__D01_01__` | `raw.isna().sum()` | 缺失统计面向刚读到的全表，与后面的有效成绩分母不同。 |
| `__D01_02__` | `raw['G3'].between(0, 20)` | 区间条件包含 0 和 20，自然排除缺失；不能对目标值随便插补。 |
| `__D01_03__` | `['max', 'min', 'mean', 'median']` | 把四个统计量一次应用在合法成绩序列上。 |
| `__D01_04__` | `bins=[0, 5, 10, 15, 21], right=False` | 分箱左闭右开；由于此前先排除超过 20 的值，最后箱用右边界 21 将 20 纳入。 |

**真实运行输出**

```text
原表行列 (43, 10)
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 43 entries, 0 to 42
Data columns (total 10 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   id           42 non-null     object 
 1   school       41 non-null     object 
 2   support      41 non-null     object 
 3   absences     41 non-null     float64
 4   study_hours  41 non-null     float64
 5   term_weeks   42 non-null     float64
 6   assigned     42 non-null     float64
 7   submitted    42 non-null     float64
 8   G2           42 non-null     float64
 9   G3           41 non-null     float64
dtypes: float64(7), object(3)
memory usage: 3.5+ KB
有效成绩最小值和最大值 0.0 20.0 各箱计数 [3, 12, 17, 8]
```

## D02 清洗、修复缺勤和构造预警标签

[完整参考实现](参考实现/D02.py)

| 空位 | 填入 | 为什么 |
| --- | --- | --- |
| `__D02_01__` | `'all'` | 只删全空行；不能误用任意字段缺失就删行。 |
| `__D02_02__` | `nonempty['G3'].between(0, 20)` | 标签要来自真实合法成绩，不能先插补 G3 再设标签。 |
| `__D02_03__` | `frame.loc[valid_abs, 'absences'].median()` | 排除不合法缺勤及缺失：不能用被污染的全列算修复中位数。 |
| `__D02_04__` | `~valid_abs` | 合法掩码取反会同时选择越界和缺失缺勤。 |
| `__D02_05__` | `frame['study_hours'].mean()` | 数值型学习时长使用当前保留人群的有效均值。 |
| `__D02_06__` | `frame[col].mode().iloc[0]` | mode 可能返回多个值，本题明确选首个给 fillna 一个标量。 |
| `__D02_07__` | `(frame['G3'] < 10).astype(int)` | 严格小于阈值才标 1，10 为 0，0 合法且为 1。 |

**真实运行输出**

```text
删除行数 3 修复缺勤数 3
标签计数 {0: 25, 1: 15}
```

## D03 衍生特征、分层划分与列预处理

[完整参考实现](参考实现/D03.py)

| 空位 | 填入 | 为什么 |
| --- | --- | --- |
| `__D03_01__` | `frame['absences'] / frame['term_weeks']` | 用正确分母把缺勤换算到每周；随卷学期周数都为正。 |
| `__D03_02__` | `frame['assigned'] - frame['submitted']` | 差值方向要用布置减提交，不能反过来。 |
| `__D03_03__` | `frame['risk']` | 标签单列，不让标签和编号混入特征。 |
| `__D03_04__` | `y` | 分层依据是类别标签，保持两边分布相近。 |
| `__D03_05__` | `StandardScaler()` | 填补在前，让标准化器看已填补的训练列。 |
| `__D03_06__` | `'ignore'` | 新类别没有训练列时不报错，不能为测试集新增列。 |
| `__D03_07__` | `fit_transform` | 训练集负责拟合所有填补量、尺度和类别。 |
| `__D03_08__` | `transform` | 测试矩阵复用训练参数，并保持相同列含义。 |

**真实运行输出**

```text
训练与测试特征矩阵 (138, 8) (46, 8)
编码后列名 ['num__weekly_absences', 'num__study_hours', 'num__unsubmitted', 'cat__school_east', 'cat__school_north', 'cat__school_south', 'cat__support_no', 'cat__support_yes']
```

## D04 双模型评估与可复用保存

[完整参考实现](参考实现/D04.py)

| 空位 | 填入 | 为什么 |
| --- | --- | --- |
| `__D04_01__` | `pipe, X_train, y_train` | 交叉验证必须拿整条管道；每个训练折单独拟合预处理。 |
| `__D04_02__` | `'recall'` | 题意优先减少漏掉高风险，评分使用正类召回率。 |
| `__D04_03__` | `X_train, y_train` | 最终拟合只接收训练集，测试集是独立评估用。 |
| `__D04_04__` | `probability` | ROC AUC 用高风险概率的排序信息，不用预测类别。 |
| `__D04_05__` | `pipe, model_file` | 保存整条管道，拥有填补、缩放、编码和模型，而不只是分类器。 |
| `__D04_06__` | `metrics['cv_recall'].idxmax()` | 取训练折均值最大；列表中逻辑回归在前，因此并列时它优先。 |

首段用临时隔离的 LightGBM 4.6.0 实跑，未安装到 `BaiQing3.10AI`。下段是此环境原样运行时的输出，明确标记练习替代模型；两者不是同一个模型。

**真实运行输出**

```text
model  cv_recall  accuracy  precision  recall     f1    auc
LogisticRegression     0.5458    0.7609     0.7000  0.7368 0.7179 0.8577
          LightGBM     0.5663    0.8043     0.7273  0.8421 0.7805 0.8421
仅按训练折召回率选中 LightGBM
```

**不加载临时隔离依赖时的真实输出**

```text
model  cv_recall  accuracy  precision  recall     f1    auc
             LogisticRegression     0.5458    0.7609     0.7000  0.7368 0.7179 0.8577
HistGradientBoosting (fallback)     0.5468    0.8043     0.7273  0.8421 0.7805 0.8509
仅按训练折召回率选中 HistGradientBoosting (fallback)
```

## D05 不平衡网格搜索与高风险名单

[完整参考实现](参考实现/D05.py)

| 空位 | 填入 | 为什么 |
| --- | --- | --- |
| `__D05_01__` | `'balanced'` | balanced 由各次模型拟合时的数据计算类别权重，无需手工复制样本。 |
| `__D05_02__` | `'average_precision'` | AP 按概率评估少数类的排序；不能把测试 AP 当调参目标。 |
| `__D05_03__` | `X_train, y_train` | 搜索只看训练数据；整条流水线进入 CV 才不会让预处理偷看验证折。 |
| `__D05_04__` | `best_estimator_` | refit=True 已用整份训练集重训最优管道。 |
| `__D05_05__` | `X_test, y_test` | 用独立测试集做事后重要性描述；降幅不是因果效应，也可能为负。 |
| `__D05_06__` | `new[cols]` | 只传给预测，不加到训练集；保留相同字段顺序。 |
| `__D05_07__` | `profiles['p_risk'] >= 0.50` | 名单阈值含等号，之后才排序与截取前五。 |

**真实运行输出**

```text
训练标签计数 {0: 82, 1: 56}
最优参数 {'model__C': 0.05, 'model__class_weight': None} 训练折 AP 0.6667 测试 AP 0.7894
达标且入名单的编号 ['N01', 'N09', 'N04', 'N03', 'N06']
```

## D06 本地量化模型报告及失败回退

[完整参考实现](参考实现/D06.py)

| 空位 | 填入 | 为什么 |
| --- | --- | --- |
| `__D06_01__` | `model_path` | model_path 要指向实际的既有 GGUF，本地 CPU 运行。 |
| `__D06_02__` | `loads` | loads 读取字符串；load 读取文件对象。额外的转换钩子会拒绝重复键。 |
| `__D06_03__` | `set(parsed) != expected` | 先比字段集合，避免多字段或缺字段造成漏检。 |
| `__D06_04__` | `type(parsed[field]) is not type(facts[field]) or parsed[field] != facts[field]` | 逐项严格核验类型与数值；例如布尔值不能冒充数字。 |
| `__D06_05__` | `parsed['action'] not in allowed` | 限制为给定建议，其他自然语言内容不当作可用结果。 |
| `__D06_06__` | `facts, ensure_ascii=False` | 序列化的是当前这位学生的事实；中文原样保留。 |
| `__D06_07__` | `create_chat_completion` | 真实调用聊天接口并保存返回值，不能用写死的句子模拟模型。 |

`invalid_json` 表示实际模型回复没有通过格式检查，不代表脚本没运行。最终报告若来自模板，会明确写 `fallback`。
[查看本次实跑的提示、原回复与最终报告](D06_实跑报告证据.json)。

**真实运行输出**

```text
报告检查 P101 invalid_json 来源 fallback
报告检查 P205 invalid_json 来源 fallback
```

# D05：参考实现。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "D_学业预警流程填空", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "D05")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

frame = pd.read_csv(os.path.join(DATA, 'campus_model.csv'))
new = pd.read_csv(os.path.join(DATA, 'screening.csv'))
nums = ['absences', 'study_hours']
cats = ['school', 'support']
cols = nums + cats
X_train, X_test, y_train, y_test = train_test_split(frame[cols], frame['risk'], test_size=0.25,
                                                   random_state=91, stratify=frame['risk'])
num = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
cat = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
pre = ColumnTransformer([('num', num, nums), ('cat', cat, cats)])
pipe = Pipeline([('pre', pre), ('model', LogisticRegression(max_iter=500))])
# balanced 由各次模型拟合时的数据计算类别权重，无需手工复制样本。
grid = {'model__C': [0.05, 0.4, 4.0], 'model__class_weight': [None, 'balanced']}
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=91)
# AP 按概率评估少数类的排序；不能把测试 AP 当调参目标。
search = GridSearchCV(pipe, grid, cv=folds, scoring='average_precision', refit=True)
# 搜索只看训练数据；整条流水线进入 CV 才不会让预处理偷看验证折。
search.fit(X_train, y_train)
# refit=True 已用整份训练集重训最优管道。
model = search.best_estimator_
pos = list(model.classes_).index(1)
ap = average_precision_score(y_test, model.predict_proba(X_test)[:, pos])
# 用独立测试集做事后重要性描述；降幅不是因果效应，也可能为负。
importance = permutation_importance(model, X_test, y_test, scoring='average_precision', n_repeats=7, random_state=91)
ranking = pd.DataFrame({'feature': cols, 'ap_drop': importance.importances_mean}).sort_values('ap_drop', ascending=False)
# 只传给预测，不加到训练集；保留相同字段顺序。
probabilities = model.predict_proba(new[cols])[:, pos]
profiles = new.assign(p_risk=probabilities)
# 名单阈值含等号，之后才排序与截取前五。
flagged = profiles.loc[profiles['p_risk'] >= 0.50].sort_values(['p_risk', 'id'], ascending=[False, True]).head(5)
grid_report = pd.DataFrame(search.cv_results_)[['param_model__C', 'param_model__class_weight', 'mean_test_score', 'rank_test_score']]
grid_report.to_csv(os.path.join(OUT, 'cv_grid.csv'), index=False)
ranking.to_csv(os.path.join(OUT, 'feature_importance.csv'), index=False)
flagged.to_csv(os.path.join(OUT, 'risk_list.csv'), index=False)
pd.DataFrame({'cv_ap': [search.best_score_], 'test_ap': [ap]}).to_csv(os.path.join(OUT, 'validation.csv'), index=False)
print('训练标签计数', y_train.value_counts().sort_index().to_dict())
print('最优参数', search.best_params_, '训练折 AP', round(search.best_score_, 4), '测试 AP', round(ap, 4))
print('达标且入名单的编号', flagged['id'].tolist())

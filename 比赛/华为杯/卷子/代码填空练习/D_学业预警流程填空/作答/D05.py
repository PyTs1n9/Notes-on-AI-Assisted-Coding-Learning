# D05：只替换带题号的空位。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))  # 随卷数据目录。
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
grid = {'model__C': [0.05, 0.4, 4.0], 'model__class_weight': [None, __D05_01__]}
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=91)
search = GridSearchCV(pipe, grid, cv=folds, scoring=__D05_02__, refit=True)
search.fit(__D05_03__)
model = search.__D05_04__
pos = list(model.classes_).index(1)
ap = average_precision_score(y_test, model.predict_proba(X_test)[:, pos])
importance = permutation_importance(model, __D05_05__, scoring='average_precision', n_repeats=7, random_state=91)
ranking = pd.DataFrame({'feature': cols, 'ap_drop': importance.importances_mean}).sort_values('ap_drop', ascending=False)
probabilities = model.predict_proba(__D05_06__)[:, pos]
profiles = new.assign(p_risk=probabilities)
flagged = profiles.loc[__D05_07__].sort_values(['p_risk', 'id'], ascending=[False, True]).head(5)
grid_report = pd.DataFrame(search.cv_results_)[['param_model__C', 'param_model__class_weight', 'mean_test_score', 'rank_test_score']]
grid_report.to_csv(os.path.join(OUT, 'cv_grid.csv'), index=False)
ranking.to_csv(os.path.join(OUT, 'feature_importance.csv'), index=False)
flagged.to_csv(os.path.join(OUT, 'risk_list.csv'), index=False)
pd.DataFrame({'cv_ap': [search.best_score_], 'test_ap': [ap]}).to_csv(os.path.join(OUT, 'validation.csv'), index=False)
print('训练标签计数', y_train.value_counts().sort_index().to_dict())
print('最优参数', search.best_params_, '训练折 AP', round(search.best_score_, 4), '测试 AP', round(ap, 4))
print('达标且入名单的编号', flagged['id'].tolist())

# D04：只替换带题号的空位。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "D04")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMClassifier
except ModuleNotFoundError as error:
    if error.name != 'lightgbm':
        raise
    second = HistGradientBoostingClassifier(max_iter=70, random_state=77)
    second_name = 'HistGradientBoosting (fallback)'
else:
    second = LGBMClassifier(n_estimators=70, random_state=77, n_jobs=1, verbosity=-1)
    second_name = 'LightGBM'

frame = pd.read_csv(os.path.join(DATA, 'campus_model.csv'), index_col='id')
num_cols = ['absences', 'study_hours']
cat_cols = ['school', 'support']
X = frame[num_cols + cat_cols]
y = frame['risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=81, stratify=y)
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
candidates = [('LogisticRegression', LogisticRegression(max_iter=500), 'baseline'), (second_name, second, 'second')]
rows = []
predictions = pd.DataFrame({'id': X_test.index, 'true': y_test.to_numpy()})
for name, estimator, filename in candidates:
    num = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    cat = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    pre = ColumnTransformer([('num', num, num_cols), ('cat', cat, cat_cols)])
    pre.set_output(transform='pandas')  # 保留列名，也方便树模型的特征检查。
    pipe = Pipeline([('pre', pre), ('model', estimator)])
    cv_scores = cross_val_score(__D04_01__, cv=folds, scoring=__D04_02__)
    pipe.fit(__D04_03__)
    pred = pipe.predict(X_test)
    probability = pipe.predict_proba(X_test)[:, list(pipe.classes_).index(1)]
    rows.append({'model': name, 'cv_recall': cv_scores.mean(), 'accuracy': accuracy_score(y_test, pred),
                 'precision': precision_score(y_test, pred, zero_division=0),
                 'recall': recall_score(y_test, pred, zero_division=0),
                 'f1': f1_score(y_test, pred, zero_division=0), 'auc': roc_auc_score(y_test, __D04_04__)})
    predictions[filename + '_label'] = pred
    predictions[filename + '_p'] = probability
    model_file = os.path.join(OUT, filename + '.joblib')
    joblib.dump(__D04_05__)
    restored = joblib.load(model_file)
    assert np.allclose(restored.predict_proba(X_test), pipe.predict_proba(X_test))
metrics = pd.DataFrame(rows)
choice = metrics.loc[__D04_06__, 'model']
metrics.to_csv(os.path.join(OUT, 'metrics.csv'), index=False)
predictions.to_csv(os.path.join(OUT, 'predictions.csv'), index=False)
pd.DataFrame({'criterion': ['training_cv_recall'], 'model': [choice]}).to_csv(os.path.join(OUT, 'model_choice.csv'), index=False)
print(metrics.round(4).to_string(index=False))
print('仅按训练折召回率选中', choice)

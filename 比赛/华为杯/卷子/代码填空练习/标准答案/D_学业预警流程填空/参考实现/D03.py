# D03：参考实现。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "D_学业预警流程填空", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "D03")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

frame = pd.read_csv(os.path.join(DATA, 'campus_model.csv'), index_col='id')
# 用正确分母把缺勤换算到每周；随卷学期周数都为正。
frame['weekly_absences'] = frame['absences'] / frame['term_weeks']
# 差值方向要用布置减提交，不能反过来。
frame['unsubmitted'] = frame['assigned'] - frame['submitted']
num_cols = ['weekly_absences', 'study_hours', 'unsubmitted']
cat_cols = ['school', 'support']
X = frame[num_cols + cat_cols]
# 标签单列，不让标签和编号混入特征。
y = frame['risk']
# 分层依据是类别标签，保持两边分布相近。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=77, stratify=y)
# 填补在前，让标准化器看已填补的训练列。
num = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                # 新类别没有训练列时不报错，不能为测试集新增列。
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
pre = ColumnTransformer([('num', num, num_cols), ('cat', cat, cat_cols)])
# 训练集负责拟合所有填补量、尺度和类别。
train_matrix = pre.fit_transform(X_train)
# 测试矩阵复用训练参数，并保持相同列含义。
test_matrix = pre.transform(X_test)
names = pre.get_feature_names_out()
pd.DataFrame(train_matrix, index=X_train.index, columns=names).assign(risk=y_train).to_csv(os.path.join(OUT, 'train_matrix.csv'))
pd.DataFrame(test_matrix, index=X_test.index, columns=names).assign(risk=y_test).to_csv(os.path.join(OUT, 'test_matrix.csv'))
print('训练与测试特征矩阵', train_matrix.shape, test_matrix.shape)
print('编码后列名', names.tolist())

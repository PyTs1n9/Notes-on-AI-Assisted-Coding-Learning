# D03：只替换带题号的空位。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "D03")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

frame = pd.read_csv(os.path.join(DATA, 'campus_model.csv'), index_col='id')
frame['weekly_absences'] = __D03_01__
frame['unsubmitted'] = __D03_02__
num_cols = ['weekly_absences', 'study_hours', 'unsubmitted']
cat_cols = ['school', 'support']
X = frame[num_cols + cat_cols]
y = __D03_03__
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=77, stratify=__D03_04__)
num = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', __D03_05__)])
cat = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown=__D03_06__, sparse_output=False))])
pre = ColumnTransformer([('num', num, num_cols), ('cat', cat, cat_cols)])
train_matrix = pre.__D03_07__(X_train)
test_matrix = pre.__D03_08__(X_test)
names = pre.get_feature_names_out()
pd.DataFrame(train_matrix, index=X_train.index, columns=names).assign(risk=y_train).to_csv(os.path.join(OUT, 'train_matrix.csv'))
pd.DataFrame(test_matrix, index=X_test.index, columns=names).assign(risk=y_test).to_csv(os.path.join(OUT, 'test_matrix.csv'))
print('训练与测试特征矩阵', train_matrix.shape, test_matrix.shape)
print('编码后列名', names.tolist())

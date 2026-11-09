import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, '..', '16_第十六课：数据泄漏与综合清洗实战', 'hour_clean.csv')
csv_path = os.path.abspath(csv_path)

df = pd.read_csv(csv_path, encoding='utf-8-sig')
y = df['cnt']

num_cols = ['temp', 'atemp', 'hum', 'windspeed']
cat_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
X = df[num_cols + cat_cols]
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)

num_pipe = Pipeline([
	('impute', SimpleImputer(strategy='median')),
	('scaler', StandardScaler())
])

cat_pipe = Pipeline([
	('impute', SimpleImputer(strategy='most_frequent')),
	('onehot', OneHotEncoder(handle_unknown='ignore'))
])

prep = ColumnTransformer([
	('num', num_pipe, num_cols),
	('cat', cat_pipe, cat_cols)
])

model_pipe = Pipeline([
	('prep', prep),
	('model', LinearRegression())
])

model_pipe.fit(X_train, y_train)
pred = model_pipe.predict(X_test)
X_test_transform = prep.transform(X_test)

mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

cat_feature_names = prep.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
all_feature_names = num_cols + cat_feature_names.tolist()

print('训练集形状', X_train.shape, y_train.shape)
print('测试集形状', X_test.shape, y_test.shape)
print('预处理后测试集形状', X_test_transform.shape)
print('最终特征名', all_feature_names)
print('MAE: ', mae)
print('R2: ', r2)

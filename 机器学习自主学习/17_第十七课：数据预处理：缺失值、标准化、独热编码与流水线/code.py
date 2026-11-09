import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

raw_df = pd.DataFrame({
	'面积': [50.0, 60.0, None, 80.0, 90.0, 100.0, 110.0, 120.0],
	'楼层': [2.0, 5.0, 3.0, None, 8.0, 10.0, 12.0, 15.0],
	'区域': ['东区', '西区', '东区', '南区', np.nan, '西区', '北区', '东区'],
	'租金': [3.0, 3.8, 3.5, 4.5, 5.0, 5.8, 6.2, 7.0]
})

X = raw_df.drop(columns=['租金'])
y = raw_df['租金']

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.25, random_state=42
)

num_features = ['面积', '楼层']
cat_features = ['区域']

num_pipe = Pipeline([
	('imputer', SimpleImputer(strategy='median')),
	('scaler', StandardScaler())
])

cat_pipe = Pipeline([
	('imputer', SimpleImputer(strategy='most_frequent')),
	('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

prep = ColumnTransformer([
	('num', num_pipe, num_features),
	('cat', cat_pipe, cat_features)
])

model_pipeline = Pipeline([
	('prep', prep),
	('model', LinearRegression())
])

model_pipeline.fit(X_train, y_train)
pred = model_pipeline.predict(X_test)
# 取出已经拟合的列转换器
fitted_prep = model_pipeline.named_steps['prep']
# 查看预处理后最终生成的特征名称
features_name = fitted_prep.get_feature_names_out()
# 单独查看测试集经过预处理后的二维数组
transformed_test = fitted_prep.transform(X_test)

# 打印原始表、切分结果和预处理结果的关键信息
print('raw shape:', raw_df.shape)
print('train shape:', X_train.shape)
print('test shape:', X_test.shape)
print('transformed test shape:', transformed_test.shape)
print('feature names:', list(features_name))
print('predictions:', [round(value, 4) for value in pred])
print('test score:', round(model_pipeline.score(X_test, y_test), 4))
import pandas as pd

s = pd.Series([90, 85, 78], index=['小明', '小红', '小刚'], name='数学')
print(s)
print("s的类型：", type(s))

df = pd.DataFrame({
	'name': ['xdj', 'xxx', 'ddd', 'jjj'],
	'math': [90, 100, 80, 30],
	'english': [28, 37, 90, 100],
	'python': [80, 90, 20, 38]
})
print(df)

print(df.head(2))
print("形状:", df.shape)
print("列名:", df.columns.tolist())
print("各列类型:")
print(df.dtypes)

df.info()
print(df.describe())
print("手算数学平均分:", (90 + 85 + 78 + 92 + 88) / 5)
print(pd.__version__)
import pandas as pd

df = pd.DataFrame({
	"电影名": ['film01', 'film02', 'film03', 'film04', 'film05'],
	"票价": [50, 80, 38, 40, 60],
	"评分": [9.7, 6.3, 7.7, 8.9, 10.0]
})

print(df.head(3))
print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
df.info()
print(df.describe())
print("手算票价平均值：", (50 + 80 + 38 + 40 + 60) / 5)

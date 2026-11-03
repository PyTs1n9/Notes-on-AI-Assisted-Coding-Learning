import pandas as pd

df = pd.read_csv('ride_audit.csv', encoding='utf-8-sig')

print(df.shape)
print(df.dtypes)
print(df.head())
print(df.isnull().sum())
print(df.isnull().sum().sum())
print(df.describe().round(2))

df = pd.DataFrame({
	'类型': df.dtypes,
	'缺失数': df.isnull().sum()
})

print(df)

print('hum,windspeed 各缺失1个，temp 缺失2个，一共4个。')

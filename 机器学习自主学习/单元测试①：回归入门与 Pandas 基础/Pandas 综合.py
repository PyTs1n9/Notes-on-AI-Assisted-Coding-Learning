import pandas as pd

df = pd.DataFrame({
	'姓名': ['甲', '乙', '丙', '丁', '戊'],
	'数学': [90, 60, 75, 88, 52],
	'英语': [80, 70, 85, 66, 95],
}, index=['s1', 's2', 's3', 's4', 's5'])

print(df)

print(df.loc['s3'])

print(df.loc['s4', '数学'])
print(df.iloc[2, 1])
print(df[df['数学'] >= 80])
print('手数行数为2行，分别是s1和s4')
print(df[(df['数学'] >= 70) & (df['英语'] >= 80)])
print('手数行数为2行，分别是s1和s3')

df.to_csv('单测1.csv', index=False, encoding='utf-8-sig')
df = pd.read_csv('单测1.csv')
print(df.shape)
print(type(df['数学']))

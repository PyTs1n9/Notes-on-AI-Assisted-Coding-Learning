import pandas as pd

df = pd.DataFrame({
	'name': ['xdj', 'djd', 'xxx', 'yyy', 'zzz'],
	'math': [90, 85, 78, 92, 88],
	'english': [76, 92, 80, 85, 79],
	'python': [88, 90, 70, 60, 55],
}, index=['a', 'b', 'c', 'd', 'e'])
print(df)

print(df['name'])
print("取一列的类型:", type(df['name']))

print(df.loc['b'])
print(df.loc['a':'c','math':'english'])
print(df.loc[df['math'] >= 85, ['name', 'math']])

print(df.iloc[1])
print(df.iloc[0:2, 1:3])
print(df.iloc[3, 2])

print(df[df['python'] >= 85])
print(df[(df['math'] >= 85) & (df['english'] >= 80)])

df.to_csv('成绩单.csv', index=False, encoding='utf-8-sig')
df2 = pd.read_csv('成绩单.csv', encoding='utf-8-sig')
print(df2)
print("读回来的形状:", df2.shape)
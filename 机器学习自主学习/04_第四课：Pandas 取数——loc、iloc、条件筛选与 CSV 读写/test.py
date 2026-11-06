import pandas as pd

df = pd.DataFrame({
	'电影名': ['film1', 'film2', 'film3', 'film4', 'film5'],
	'票房': [100, 30, 200, 50, 40],
	'评分': [8.8, 9.2, 6.7, 7.9, 10.0],
	'上映天数': [5, 3, 2, 1, 8],
}, index=['m1', 'm2', 'm3', 'm4', 'm5'])
print(df)

df.to_csv('票房.csv', index=True, encoding='utf-8-sig')
df = pd.read_csv('票房.csv', index_col=0)
print(df.shape)

print(type(df['票房']))
print(df)
print(df.loc['m3', ['电影名', '票房']])

print(df.iloc[3, 1])
print('手查对照：第 4 行是 m4(film4)，第 1 列是票房 → 50')
print(df[df['评分'] >= 8.5])
print('手数行数：3（我数的是 m1=8.8、m2=9.2、m5=10.0）')

print(df[(df['票房'] >= 50) & (df['评分'] >= 8)])
print('手数行数：1（只有 m1：票房 100、评分 8.8）')

import pandas as pd

df = pd.read_csv('ride_feat.csv', encoding='utf-8-sig')

print(df.shape)
print(df['dteday'].dtype)

df['dteday'] = pd.to_datetime(df['dteday'])
print(df['dteday'].dtype)
df['year'] = df['dteday'].dt.year
df['month'] = df['dteday'].dt.month
df['weekdays'] = df['dteday'].dt.weekday
print(df['weekday'].head(3))

df['season'] = df['season'].astype('str')
print(df['season'].dtype)

df = pd.get_dummies(df, columns=['season'])
print(df.shape)
print(list(df.columns))

y = df['cnt']
X = df.drop(columns=['cnt','dteday'])
print(X.shape)
print(y.shape)
print(y.head(3))

print('因为不转的话是object类型，无法识别')
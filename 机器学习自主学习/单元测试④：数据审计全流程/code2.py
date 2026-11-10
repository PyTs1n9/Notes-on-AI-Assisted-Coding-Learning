import pandas as pd

df = pd.read_csv('rental.csv', encoding='utf-8-sig')
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.weekday

cat_cols = ['season', 'holiday']
df = pd.get_dummies(df, columns=cat_cols, dtype=int)
print(df.shape)
print(df.columns)

print((df['casual']+df['registered'] == df['total']).all())
print((df['casual']+df['registered'] == df['total']).sum())


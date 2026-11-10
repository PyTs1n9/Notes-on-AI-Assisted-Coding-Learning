import pandas as pd

df = pd.read_csv('clinic.csv', encoding='utf-8-sig')

print(df.shape)
print(df.dtypes)
print(df.head())
print(df.isnull().sum())
print(df.isnull().sum().sum())
print(df.duplicated().sum())
df = df.drop_duplicates()
print(df.shape)

q1 = df['temperature'].quantile(0.25)
q3 = df['temperature'].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
print(f'lower {lower:.2f}')
print(f'upper {upper:.2f}')
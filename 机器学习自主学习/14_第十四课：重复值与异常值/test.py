import pandas as pd

df = pd.read_csv('ride_dup_outlier.csv', encoding='utf-8-sig')
print(df.shape)
print(df.duplicated().sum())
df = df.drop_duplicates()
print(df.shape)
q1 = df['windspeed'].quantile(0.25)
q3 = df['windspeed'].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
print(f'Q1={q1:.4f} Q3={q3:.4f} IQR={iqr:.4f} Lower={lower:.4f} Upper={upper:.4f}')
out = df[(df['windspeed'] < lower) | (df['windspeed'] > upper)]
print(len(out))
print(out)
q1c = df['casual'].quantile(0.25)
q3c = df['casual'].quantile(0.75)
iqrc = q3c - q1c
lowerc = q1c - 1.5 * iqrc
upperc = q3c + 1.5 * iqrc
outc = df[(df['casual'] < lowerc) | (df['casual'] > upperc)]
print(len(outc))
print('可以不用删，出现极端风速是正常现象，出现重复的风速就不正常了，数值上对即可，不算错')

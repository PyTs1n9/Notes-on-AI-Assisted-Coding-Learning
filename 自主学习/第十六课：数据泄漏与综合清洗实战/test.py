import pandas as pd

csv_list = ['hour.csv', 'day.csv']
for csv in csv_list:
	df = pd.read_csv(csv, encoding='utf-8-sig')
	print(f"{csv} {df.shape}")
	print(f'{csv} {df.duplicated().sum().sum()}')
	print(f'{csv} {df.duplicated().sum()}')
	print(f'{csv} {len((df["cnt"] == df["casual"] + df["registered"]))}')
	corr = df.corr(numeric_only=True)['cnt']
	print(f'{csv} casual：{round(corr["casual"], 6)}')
	print(f'{csv} registered：{round(corr["registered"], 6)}')
	df = df.drop_duplicates('windspeed')
	Q1 = df['windspeed'].quantile(0.25)
	Q3 = df['windspeed'].quantile(0.75)
	IQR = Q3 - Q1
	lower = Q1 - 1.5 * IQR
	upper = Q3 + 1.5 * IQR
	out = df[(df['windspeed'] < lower) | (df['windspeed'] > upper)]

	df = df.drop(columns=['instant', 'dteday', 'casual', 'registered'])
	print(f"{csv} {df.shape}")
	print()

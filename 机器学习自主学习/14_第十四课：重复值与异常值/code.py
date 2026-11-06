# ============ 第十四课演示：给真实单车租赁数据找"重复行"和"异常值" ============
import pandas as pd

df = pd.read_csv('day.csv', encoding='utf-8-sig')

print(df.duplicated().sum())
print(df.shape)
df = df.drop_duplicates()
print(df.shape)

# ---------- 第 3 步：用 IQR 给"风速 windspeed"算异常上下限 ----------
q1 = df['windspeed'].quantile(0.25)
q3 = df['windspeed'].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
print(f'Q1={q1:.4f} Q3={q3:.4f} IQR={iqr:.4f} lower={lower:.4f} upper={upper:.4f}')

# ---------- 第 4 步：按上下限把异常候选行捞出来 ----------
out = df[(df['windspeed'] < lower) | (df['windspeed'] > upper)]
print(len(out))  # 异常候选行数
print(out['windspeed'])  # 每一行的风速（左边的数字是行号）

# ---------- 第 5 步：坑演示：0/1 列套 IQR 会误伤 ----------
q1h = df['holiday'].quantile(0.25)
q3h = df['holiday'].quantile(0.75)
ih = q3h - q1h
outh = df[(df['holiday'] < q1h - 1.5 * ih) | (df['holiday'] > q3h + 1.5 * ih)]
print(f'holiday 的 IQR={ih:.1f}，被判异常的行数={len(outh)}') 
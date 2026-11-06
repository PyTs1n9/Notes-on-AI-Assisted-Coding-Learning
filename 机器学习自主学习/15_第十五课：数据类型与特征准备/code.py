# ============ 第十五课演示：给租赁数据"转日期 → 拆特征 → 出 X/y" ============
import pandas as pd

# ---------- 第 1 步：dteday 现在是文本，先转成真正的日期 ----------
df = pd.read_csv('day.csv', encoding='utf-8-sig')
print(df['dteday'].dtype)
df['dteday'] = pd.to_datetime(df['dteday'])
print(df['dteday'].dtype)

# ---------- 第 2 步：从日期里拆出年、月、星期几（衍生特征） ----------
df['year'] = df['dteday'].dt.year
df['month'] = df['dteday'].dt.month
df['weekday'] = df['dteday'].dt.weekday
print(df['weekday'].head(3))

# ---------- 第 3 步：类别列转成文本，再拆成一排 0/1 列 ---------
df['season'] = df['season'].astype('str')
df = pd.get_dummies(df, columns=['season'])
print(df.shape)
print(list(df.columns))

# ---------- 第 4 步：出特征表 X 和标签 y ----------
y = df['cnt']
X = df.drop(columns=['cnt', 'dteday'])
print(X.shape)
print(y.shape)
print(y.head(3))

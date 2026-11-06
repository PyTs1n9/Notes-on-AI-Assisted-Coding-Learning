# ============ 第十六课演示：给租赁数据做"泄漏体检" ============
import pandas as pd

df = pd.read_csv('ride_feat.csv', encoding='utf-8-sig')

# ---------- 第 1 步：泄漏体检——标签是不是被"加"出来的 ----------
same = df['cnt'] == df['casual'] + df['registered']
print(same.all())
print(int(same.sum()))

# ---------- 第 2 步：相关系数——谁和标签最像 ----------
corr = df.corr(numeric_only=True)['cnt']
print(corr.sort_values(ascending=False).head(5))

# ---------- 第 3 步：把"答案的成分"从特征里删掉 ----------
X = df.drop(columns=['cnt', 'casual', 'registered', 'instant', 'dteday'])
print(X.shape) 
import pandas as pd  # 导入 pandas，完成日期、编码与泄漏审查

df = pd.read_csv('rental.csv', encoding='utf-8-sig')  # 读取租赁记录
print('原表形状:', df.shape)  # 输出原始行列数
print('原日期类型:', df['date'].dtype)  # 明确输出日期列转换前的类型
print('缺失总数:', df.isnull().sum().sum())  # 输出整表缺失总数

df['date'] = pd.to_datetime(df['date'])  # 先把日期文本转换成日期类型
print('转换后日期类型:', df['date'].dtype)  # 核对类型转换结果
df['year'] = df['date'].dt.year  # 提取年份到新列
df['month'] = df['date'].dt.month  # 提取月份到新列
df['weekday'] = df['date'].dt.weekday  # 提取星期，周一为 0
df['season'] = df['season'].astype('str')  # 按题目要求把季节代码转成文本
print('季节转换后类型:', df['season'].dtype)  # 核对类别代码已转为文本
df = pd.get_dummies(df, columns=['season'], dtype=int)  # 只拆题目指定的 season，保留原 holiday
print('编码后形状:', df.shape)  # 核对编码后的尺寸
print('编码后列名:', list(df.columns))  # 明确打印所有列名

same = df['casual'] + df['registered'] == df['total']  # 逐行检验分项和总量的恒等关系
print('恒等关系全部成立:', same.all())  # 输出是否全表成立
print('恒等关系成立行数:', same.sum())  # 输出成立的行数
corr = df.corr(numeric_only=True)['total']  # 只计算数值列的相关矩阵并取 total 列
print('casual 与 total 的相关系数:', round(corr['casual'], 4))  # 输出分项相关性
print('registered 与 total 的相关系数:', round(corr['registered'], 4))  # 输出另一分项相关性

y = df['total']  # 单独保留目标列作为标签
X = df.drop(columns=['record_id', 'date', 'casual', 'registered', 'total'])  # 从特征表中移除编号、原日期、泄漏分项及标签
print('X 形状:', X.shape)  # 输出特征表尺寸
print('y 形状:', y.shape)  # 输出标签尺寸
print('X 列名:', list(X.columns))  # 核对特征列，确认不含泄漏列和目标列
print('y 列名:', y.name)  # Series 没有 columns，用 name 核对标签名称
print('X/y 行索引一致:', X.index.equals(y.index))  # 核对特征与标签行顺序对应
print('解释：本题按事前预测租赁总量理解，casual 和 registered 是事后统计分项，预测时不能使用；它们相加直接得到 total，会把答案带进特征。相关性只是线索，须结合恒等关系和字段可用时点判断。')  # 给出完整泄漏解释
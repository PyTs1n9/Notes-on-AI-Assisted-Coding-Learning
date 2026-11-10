import pandas as pd  # 导入 pandas，完成读表与审计

df = pd.read_csv('clinic.csv', encoding='utf-8-sig')  # 读取门诊记录
print('原表形状:', df.shape)  # 输出原始行列数
print('列类型:')  # 标明下面输出的是各列类型
print(df.dtypes)  # 输出各列类型
print('前几行:')  # 标明样例区域
print(df.head())  # 输出前几行记录
missing = df.isnull().sum()  # 统计原表逐列缺失数
print('逐列缺失:')  # 标明缺失统计区域
print(missing)  # 输出每列缺失数量
print('缺失总数:', missing.sum())  # 汇总所有列的缺失数量
print('整行重复数:', df.duplicated().sum())  # 统计整行重复数量

df = df.drop_duplicates()  # 接住删重后返回的新表
print('删重后形状:', df.shape)  # 核对删重后的行列数
q1 = df['temperature'].quantile(0.25)  # 计算第一四分位数，自动跳过缺失
q3 = df['temperature'].quantile(0.75)  # 计算第三四分位数
iqr = q3 - q1  # 计算四分位距
lower = q1 - 1.5 * iqr  # 计算下界
upper = q3 + 1.5 * iqr  # 计算上界
print('Q1:', round(q1, 2))  # 输出第一四分位数
print('Q3:', round(q3, 2))  # 输出第三四分位数
print('IQR:', round(iqr, 2))  # 输出四分位距
print('下界:', round(lower, 2))  # 输出下界
print('上界:', round(upper, 2))  # 输出上界
out = df[(df['temperature'] < lower) | (df['temperature'] > upper)]  # 用“或”筛出两侧越界候选
print('候选行数:', len(out))  # 输出候选行数
print(out)  # 即便候选为空，也明确输出候选表

audit = pd.DataFrame({'类型': df.dtypes, '缺失数': df.isnull().sum()})  # 按字段名对齐生成审计表
print('审计表:')  # 标明下面是审计表
print(audit)  # 输出逐字段的类型和缺失数
print('审计表形状:', audit.shape)  # 核对审计表尺寸
print('业务判断：本次温度 IQR 候选共', len(out), '行，无需按温度候选删除记录；缺失温度另作缺失审查。即使出现候选，也应核对原始记录和业务情况，不能仅凭越界认定录错。')  # 引用实测数量说明处理判断
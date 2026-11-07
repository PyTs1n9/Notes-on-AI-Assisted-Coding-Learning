import pandas as pd  # 导入 pandas，用于读表、清洗和保存 CSV

# 保存两张表的审计结果，最后写入 data_profile.csv
profile_rows = []  # 创建列表，保存 hour 和 day 各自的一行审计结果

# 保存两张表的异常候选行，最后写入 outlier_candidates.csv
outlier_frames = []  # 创建列表，保存两张表的异常候选行

# 依次处理 hour.csv 和 day.csv
for file_name in ['hour.csv', 'day.csv']:
    # 读取当前 CSV 文件
    df = pd.read_csv(file_name, encoding='utf-8-sig')

    # 从文件名中取出 hour 或 day，作为产物中的文件标识
    source_name = file_name.split('.')[0]

    # 记录原始表的行数和列数
    original_rows = df.shape[0]
    original_columns = df.shape[1]

    # 统计整张表的缺失值总数
    missing_total = int(df.isnull().sum().sum())

    # 统计重复行数量
    duplicate_rows = int(df.duplicated().sum())

    # 删除完全重复的行
    df = df.drop_duplicates()

    # 验证 cnt 是否等于 casual 加 registered
    leakage_same = df['cnt'] == df['casual'] + df['registered']

    # 统计泄漏恒等关系成立的行数
    leakage_rows = int(leakage_same.sum())

    # 计算数值列的相关系数矩阵，并取出 cnt 这一列
    corr = df.corr(numeric_only=True)['cnt']

    # 打印本表的关键验收结果
    print(f'{source_name} shape:', (original_rows, original_columns))
    print(f'{source_name} missing:', missing_total)
    print(f'{source_name} duplicates:', duplicate_rows)
    print(f'{source_name} leakage rows:', leakage_rows)
    print(f'{source_name} casual corr:', round(corr['casual'], 6))
    print(f'{source_name} registered corr:', round(corr['registered'], 6))

    # 计算 windspeed 的第一四分位数
    q1 = df['windspeed'].quantile(0.25)

    # 计算 windspeed 的第三四分位数
    q3 = df['windspeed'].quantile(0.75)

    # 计算四分位距
    iqr = q3 - q1

    # 计算 IQR 下界
    lower = q1 - 1.5 * iqr

    # 计算 IQR 上界
    upper = q3 + 1.5 * iqr

    # 筛选 windspeed 的异常候选行
    outlier_df = df[(df['windspeed'] < lower) | (df['windspeed'] > upper)].copy()

    # 添加候选行的来源文件列
    outlier_df['源文件'] = source_name

    # 保存当前表的候选行，后面统一拼接
    outlier_frames.append(outlier_df)

    # 删除行号列、日期列和泄漏列，但保留 cnt 标签列
    clean_df = df.drop(columns=['instant', 'dteday', 'casual', 'registered'])

    # 打印清洗后的表形状
    print(f'{source_name}_clean shape:', clean_df.shape)

    # 保存当前表的清洗结果，不保存 pandas 行索引
    clean_df.to_csv(f'{source_name}_clean.csv', index=False, encoding='utf-8-sig')

    # 保存当前表的一行审计结果
    profile_rows.append({
        '文件': source_name,
        '行数': original_rows,
        '列数': original_columns,
        '缺失总数': missing_total,
        '重复行数': duplicate_rows,
        '泄漏成立行数': leakage_rows
    })

# 把 hour 和 day 的审计结果组成两行表
profile_df = pd.DataFrame(profile_rows)

# 保存审计汇总表，不保存 pandas 行索引
profile_df.to_csv('data_profile.csv', index=False, encoding='utf-8-sig')

# 把两张表的异常候选行按行拼接
outlier_candidates = pd.concat(outlier_frames, ignore_index=True)

# 保存异常候选行汇总表，不保存 pandas 行索引
outlier_candidates.to_csv('outlier_candidates.csv', index=False, encoding='utf-8-sig')

# 打印两个汇总产物的形状，便于验收
print('data_profile shape:', profile_df.shape)
print('outlier_candidates rows:', outlier_candidates.shape[0])
print('outlier source counts:')
print(outlier_candidates['源文件'].value_counts().sort_index())

# 业务结论：casual 和 registered 是 cnt 的组成部分，预测时不能作为可用特征
print('结论：casual 和 registered 是 cnt 的组成部分，预测时不能作为特征。')

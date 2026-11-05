import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# income 列存在极端值 98000，income 与 age 均有缺失
df = pd.DataFrame({
    'income': [5200, 4800, np.nan, 5100, 98000, np.nan, 4900],
    'age': [25, 31, 28, np.nan, 45, 38, 29],
})

# 1. 统计每一列的缺失个数
print(df.isnull().sum())

# 2. 创建填充器：要求填充值不受极端值 98000 的影响
imputer = SimpleImputer(strategy='median')

# 3. 拟合并转换，重新还原为 DataFrame
df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
print(df_filled.round(1))

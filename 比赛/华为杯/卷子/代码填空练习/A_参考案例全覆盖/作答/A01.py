# A01：只替换带题号的空位。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "A01")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import pandas as pd

frame = pd.read_csv(os.path.join(DATA, 'processed_sentiment_analysis.csv'))
missing = __A01_01__
hit = __A01_02__
chosen = frame.loc[__A01_03__].copy()
punctuation = '，。；！（）!?+'
delete_table = str.maketrans('', '', punctuation)
chosen['clean_comment'] = __A01_04__.str.strip()
missing.rename_axis('column').reset_index(name='missing').to_csv(os.path.join(OUT, 'missing.csv'), index=False)
chosen.to_csv(os.path.join(OUT, 'chosen.csv'), index=False, encoding='utf-8-sig')
print('逐列缺失', missing.to_dict())
print('通过筛选的 ID', chosen['id'].tolist())
print('清洗文本', chosen['clean_comment'].tolist())

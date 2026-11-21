# A01：参考实现。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "A_参考案例全覆盖", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "A01")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import pandas as pd

frame = pd.read_csv(os.path.join(DATA, 'processed_sentiment_analysis.csv'))
# 先在原表统计每列，不能先删缺失行再统计。
missing = frame.isna().sum()
# 关闭正则使加号是字面字符；缺失值应返回 False。
hit = frame['comment'].str.contains('物流+', regex=False, na=False)
# 两条条件需要同时成立；每条 Series 条件配合按元素逻辑。
chosen = frame.loc[(frame['label'] == '正面') & hit].copy()
punctuation = '，。；！（）!?+'
delete_table = str.maketrans('', '', punctuation)
# 此处的 str 是 Pandas 字符串访问器；转换表第三个参数指定要删除的字符。
chosen['clean_comment'] = chosen['comment'].str.translate(delete_table).str.strip()
missing.rename_axis('column').reset_index(name='missing').to_csv(os.path.join(OUT, 'missing.csv'), index=False)
chosen.to_csv(os.path.join(OUT, 'chosen.csv'), index=False, encoding='utf-8-sig')
print('逐列缺失', missing.to_dict())
print('通过筛选的 ID', chosen['id'].tolist())
print('清洗文本', chosen['clean_comment'].tolist())

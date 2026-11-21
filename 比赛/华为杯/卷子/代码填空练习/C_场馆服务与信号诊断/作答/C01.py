# C01：只替换带题号的空位。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "C01")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

marks = '，。！？?!'
mapping = str.maketrans(marks, ' ' * len(marks))

def segment(text):
    keep = []
    for word in jieba.cut(text.translate(mapping)):
        if word.strip() and word != '请问':
            keep.append(word)
    return ' '.join(keep)

faq = pd.read_csv(os.path.join(DATA, 'faq.csv'))
queries = pd.read_csv(os.path.join(DATA, 'questions.csv'))
vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)
catalog = vectorizer.__C01_01__(faq['title'].map(segment))
lookup = vectorizer.__C01_02__(queries['query'].map(segment))
scores = cosine_similarity(__C01_03__)
results = []
for i in range(len(queries)):
    best = int(__C01_04__)
    score = float(scores[i, best])
    if __C01_05__:
        faq_id = ''
        answer = '转人工'
    else:
        faq_id = faq.at[best, 'faq_id']
        answer = faq.at[best, 'answer']
    results.append({'id': queries.at[i, 'id'], 'faq_id': faq_id, 'score': score, 'answer': answer})
pd.DataFrame(results).to_csv(os.path.join(OUT, 'faq_matches.csv'), index=False, encoding='utf-8-sig')
print('各问题命中或转人工', [row['faq_id'] or row['answer'] for row in results])
print('各问题最高相似度', [round(row['score'], 4) for row in results])

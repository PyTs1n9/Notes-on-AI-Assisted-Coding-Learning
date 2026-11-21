# C01：参考实现。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "C_场馆服务与信号诊断", "data"))  # 随卷数据目录。
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
# 知识库文本定义词表与权重，查询不参与学习。
catalog = vectorizer.fit_transform(faq['title'].map(segment))
# 复用同一个向量器，才有一致的列含义。
lookup = vectorizer.transform(queries['query'].map(segment))
# 查询传第一位，得到一行一个问题、一列一个 FAQ 的分数。
scores = cosine_similarity(lookup, catalog)
results = []
for i in range(len(queries)):
    # 用当前问题的一行取最高分索引；最大值并列时只拿首次。
    best = int(np.argmax(scores[i]))
    score = float(scores[i, best])
    # 全零稀疏向量或低于阈值都拒答，避免全零行被取成第一条。
    if lookup[i].nnz == 0 or score < 0.35:
        faq_id = ''
        answer = '转人工'
    else:
        faq_id = faq.at[best, 'faq_id']
        answer = faq.at[best, 'answer']
    results.append({'id': queries.at[i, 'id'], 'faq_id': faq_id, 'score': score, 'answer': answer})
pd.DataFrame(results).to_csv(os.path.join(OUT, 'faq_matches.csv'), index=False, encoding='utf-8-sig')
print('各问题命中或转人工', [row['faq_id'] or row['answer'] for row in results])
print('各问题最高相似度', [round(row['score'], 4) for row in results])

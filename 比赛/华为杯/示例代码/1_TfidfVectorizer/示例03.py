# 加上词组区分相反短语：词完全相同但顺序不同，词组特征能帮忙区分「取消后恢复」与「恢复后取消」。
# 用于比较向量差别。
import numpy as np
# 提供单词与词组的权重。
from sklearn.feature_extraction.text import TfidfVectorizer

# 两条操作流只交换了顺序。
texts = ['取消 恢复', '恢复 取消']

# 默认只统计单词。
single = TfidfVectorizer(tokenizer=str.split, token_pattern=None)

# 加入相邻双词。ngram_range表示一次连续考虑几个词，ngram_range=(1, 2) 既考虑 1 个词，也考虑连续的 2 个词
combo = TfidfVectorizer(tokenizer=str.split, token_pattern=None, ngram_range=(1, 2))

# 小样本可以展开检查。
one = single.fit_transform(texts).toarray()

# 在另一组列上表示相同输入。
two = combo.fit_transform(texts).toarray()

# 词顺序没有被单词模式保留。np.allclose() 用来判断两个数组中的数字是否近似相等
print('只有单词时两条相同', np.allclose(one[0], one[1]))

# 双词模式区分了顺序。
print('加入词组后两条相同', np.allclose(two[0], two[1]))

# 找到两种顺序各自的特征列。
print('词组模式列名', combo.get_feature_names_out().tolist())

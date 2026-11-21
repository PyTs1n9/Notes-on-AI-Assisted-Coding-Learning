# 反馈分词后滤停用词：把一条简短反馈变成有意义的词序列；保留有意义的单字，不按字数过滤。

import jieba  # 提供分词器。

text = '今天 的 雨 影响配送'  # 有功能词，也有重要单字“雨”。
stopwords = {'今天', '的'}  # 用集合记录这次不需要的词。
tokens = []  # 保存有用词，顺序与原文相同。
for word in jieba.cut(text, cut_all=False):  # 逐个读取精确分词结果。
    if word.strip() and word not in stopwords:  # 不收空白、停用词；不丢掉单字。
        tokens.append(word)  # 把符合条件的词放进列表。
print('可用于特征提取的词', tokens)  # 这一步还没有计算权重。

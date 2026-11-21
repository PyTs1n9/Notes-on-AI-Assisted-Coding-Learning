# 精确分词和搜索索引：索引既希望命中整词，也希望能找到更短的词。比较两种模式是否提供不同的粒度。

import jieba  # 用于切分中文。

sentence = '南京市长江大桥恢复通行'  # 一条用于建索引的短公告。
normal = list(jieba.cut(sentence, cut_all=False))  # 精确模式：先得到普通词序列。
search = list(jieba.cut_for_search(sentence))  # 搜索模式：对部分长词加入更短的索引词。
print('精确分词', normal)  # 对照正常阅读粒度。
print('搜索索引', search)  # 找新出现的细粒度词。

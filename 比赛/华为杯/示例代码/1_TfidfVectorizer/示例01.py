# 看出常用模板词权重：对比工单常见词与只出现一次的关键词，而不是光看词频。
# 把词条转换为带权重的列。
from sklearn.feature_extraction.text import TfidfVectorizer

# 空格代表已经切成词，不需要再次分词。
docs = ['服务 维修', '服务 发票', '服务 退订']

# 自己定义拆词方式。
vec = TfidfVectorizer(tokenizer=str.split, token_pattern=None)

# 先学词表与权重，再转换这三条文本。
matrix = vec.fit_transform(docs)

# 取出与矩阵列一一对应的词名。
words = list(vec.get_feature_names_out())
print(words)
print(vec.get_feature_names_out())

# 仅把第一条小向量转成常见数组，便于观察。
first = matrix[0].toarray()[0]

# 跨文档常见词。
print('第一条的服务权重', round(first[words.index('服务')], 4))
# 本条特有的关键词。
print('第一条的维修权重', round(first[words.index('维修')], 4))

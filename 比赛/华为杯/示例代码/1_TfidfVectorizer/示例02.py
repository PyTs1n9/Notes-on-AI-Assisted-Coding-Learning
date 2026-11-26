# 新文本遇到单字和未知词：复用训练词表，区分「命中单字」和「完全没有训练词」。
# 提取文本特征。
from sklearn.feature_extraction.text import TfidfVectorizer

# 部分有意义的词只有一个字。
training = ['停 电 电梯', '恢复 门禁', '停 门禁']

# 第一条有一个已知词；第二条没有。
queries = ['停 大厅', '银河 光谱']

# 不用默认的两字符规则。
vec = TfidfVectorizer(tokenizer=str.split, token_pattern=None)

# 训练文本定义列。
train_matrix = vec.fit_transform(training)

# 复用列，遇到新词也不重新编号。
query_matrix = vec.transform(queries)

# 确认单字保留。
print('训练列', vec.get_feature_names_out().tolist())

# 列数必须一致。
print('训练宽度、查询宽度', train_matrix.shape[1], query_matrix.shape[1])

# 全零可作为拒判依据。getnnz() 是 get number of non-zeros
print('两条查询的非零特征数', query_matrix.getnnz(axis=1).tolist())

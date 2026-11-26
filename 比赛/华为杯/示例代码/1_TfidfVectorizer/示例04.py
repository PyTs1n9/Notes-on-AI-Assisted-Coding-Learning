from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

# 训练数据：一篇文章一行
texts = [
    "我喜欢猫",
    "我喜欢狗",
    "猫很可爱"
]

# 分词
texts_cut = [" ".join(jieba.cut(text)) for text in texts]
print(texts_cut)
# 创建工具
vec = TfidfVectorizer()

# 学习词汇并转换成数字
X = vec.fit_transform(texts_cut)

# 查看所有识别出的词
print(vec.get_feature_names_out())
print(X.toarray())

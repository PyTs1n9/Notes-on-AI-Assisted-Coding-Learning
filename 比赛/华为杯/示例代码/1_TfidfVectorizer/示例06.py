import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

texts = [
	"Python 可以用于数据分析和自动化办公",
	"机器学习可以用于垃圾邮件分类",
	"TF-IDF 可以把文本转换成数字向量",
	"余弦相似度可以计算两段文本的相似程度",
	"深度学习可以用于图像识别和语音识别",
]

def tokenize(text):
	return [word for word in jieba.cut(text) if word.isalnum()]

vectorizer = TfidfVectorizer(
	tokenizer=tokenize,  # 使用你定义的 tokenize 函数来分词
	token_pattern=None,  # 关闭默认分词规则
)
document_vectors = vectorizer.fit_transform(texts)


# 搜索：转换查询 → 计算相似度 → 按分数排序
def search(query, top_k=3):
	query_vector = vectorizer.transform([query])

	scores = cosine_similarity(
		query_vector, document_vectors
	)[0]

	indices = scores.argsort()[::-1][:top_k]

	return [
		(float(scores[i]), texts[i])
		for i in indices
		if scores[i] > 0
	]


print(vectorizer.get_feature_names_out())
# 在终端交互搜索
while True:
	query = input("\n请输入搜索内容（输入 q 退出）：").strip()

	if query.lower() == "q":
		break
	if not query:
		continue

	results = search(query)

	if not results:
		print("没有找到匹配文本")
		continue

	for rank, (score, text) in enumerate(results, start=1):
		print(f"{rank}. 相似度 {score:.3f} | {text}")

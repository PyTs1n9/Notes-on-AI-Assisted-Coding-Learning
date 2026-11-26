from sklearn.feature_extraction.text import TfidfVectorizer

texts = [
	"苹果 香蕉 苹果",
	"香蕉 橙子",
	"苹果 电脑",
]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)
print(vectorizer.get_feature_names_out())
print(X.toarray().round(3))

new_text = ['西瓜 葡萄']
new_X = vectorizer.transform(new_text)
print(new_X.toarray().round(3))

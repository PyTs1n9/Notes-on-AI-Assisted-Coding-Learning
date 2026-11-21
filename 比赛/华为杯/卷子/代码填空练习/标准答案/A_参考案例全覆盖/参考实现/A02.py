# A02：参考实现。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "A_参考案例全覆盖", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "A02")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

with open(os.path.join(DATA, 'stopwords.txt'), encoding='utf-8') as file:
    stopwords = set(file.read().splitlines())

def segment(text):
    kept = []
    # 精确模式把词按正常粒度切开；这里用兼容旧版本的 cut。
    for word in jieba.cut(text, cut_all=False):
        # 先排除纯空白，再过滤停用词；没有过滤单字。
        if word.strip() and word not in stopwords:
            kept.append(word)
    return ' '.join(kept)

news = pd.read_csv(os.path.join(DATA, 'News_data.csv'))
words = news['title'].map(segment)
labels = news['label']
# 分层依据是真实类别，不是文本或向量。
X_train, X_test, y_train, y_test = train_test_split(words, labels, test_size=0.25, random_state=29, stratify=labels)
vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)
# 训练文本负责学习词表和词权重并得到训练矩阵。
X_train_tfidf = vectorizer.fit_transform(X_train)
# 测试集只转换，列数及列含义沿用训练词表。
X_test_tfidf = vectorizer.transform(X_test)
model = LogisticRegression(max_iter=500)
# 用训练特征和对齐的训练标签拟合模型；不是测试数据。
model.fit(X_train_tfidf, y_train)
pred = model.predict(X_test_tfidf)
report = classification_report(y_test, pred, zero_division=0)
with open(os.path.join(DATA, 'query.txt'), encoding='utf-8') as file:
    query = file.read()
# 预测接收一批文本；单条也放进列表并走相同分词。
new_vector = vectorizer.transform([segment(query)])
new_label = model.predict(new_vector)[0]
print(report)
print('测试集准确率', round(accuracy_score(y_test, pred), 4))
print('新文本类别', new_label)
with open(os.path.join(OUT, 'report.txt'), 'w', encoding='utf-8') as file:
    file.write(report)
pd.DataFrame({'true': y_test.to_numpy(), 'pred': pred}).to_csv(os.path.join(OUT, 'test_predictions.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame({'text': [query], 'pred': [new_label]}).to_csv(os.path.join(OUT, 'query_prediction.csv'), index=False, encoding='utf-8-sig')

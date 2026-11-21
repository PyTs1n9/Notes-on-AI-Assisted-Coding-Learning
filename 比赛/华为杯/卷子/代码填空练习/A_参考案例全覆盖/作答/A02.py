# A02：只替换带题号的空位。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))  # 随卷数据目录。
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
    for word in __A02_01__:
        if __A02_02__:
            kept.append(word)
    return ' '.join(kept)

news = pd.read_csv(os.path.join(DATA, 'News_data.csv'))
words = news['title'].map(segment)
labels = news['label']
X_train, X_test, y_train, y_test = train_test_split(words, labels, test_size=0.25, random_state=29, stratify=__A02_03__)
vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)
X_train_tfidf = vectorizer.__A02_04__(X_train)
X_test_tfidf = vectorizer.__A02_05__(X_test)
model = LogisticRegression(max_iter=500)
model.fit(__A02_06__)
pred = model.predict(X_test_tfidf)
report = classification_report(y_test, pred, zero_division=0)
with open(os.path.join(DATA, 'query.txt'), encoding='utf-8') as file:
    query = file.read()
new_vector = vectorizer.transform(__A02_07__)
new_label = model.predict(new_vector)[0]
print(report)
print('测试集准确率', round(accuracy_score(y_test, pred), 4))
print('新文本类别', new_label)
with open(os.path.join(OUT, 'report.txt'), 'w', encoding='utf-8') as file:
    file.write(report)
pd.DataFrame({'true': y_test.to_numpy(), 'pred': pred}).to_csv(os.path.join(OUT, 'test_predictions.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame({'text': [query], 'pred': [new_label]}).to_csv(os.path.join(OUT, 'query_prediction.csv'), index=False, encoding='utf-8-sig')

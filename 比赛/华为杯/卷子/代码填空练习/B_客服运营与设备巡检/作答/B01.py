# B01：只替换带题号的空位。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "B01")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

tickets = pd.read_csv(os.path.join(DATA, 'tickets.csv'))
web = tickets['channel'].str.contains(__B01_01__)
selected = tickets.loc[web & tickets['body'].notna()].copy()
punctuation = '，。？！：?!,:'
translate_table = str.maketrans(punctuation, __B01_02__)

def prepare(text):
    keep = []
    for word in jieba.cut(__B01_03__):
        if word.strip() and word != '请问':
            keep.append(word)
    return ' '.join(keep)

X = selected['body'].map(prepare)
y = selected['priority']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=47, stratify=y)
tfidf = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)
train_vec = tfidf.fit_transform(X_train)
test_vec = tfidf.__B01_04__(X_test)
model = LogisticRegression(max_iter=500)
model.fit(train_vec, y_train)
with open(os.path.join(DATA, 'ticket_query.txt'), encoding='utf-8') as file:
    query = prepare(file.read())
probabilities = model.predict_proba(tfidf.transform([query]))
urgent_col = __B01_05__
urgent = float(probabilities[0, urgent_col])
result = '紧急' if __B01_06__ else '普通'
test_accuracy = accuracy_score(y_test, model.predict(test_vec))
pd.DataFrame({'p_urgent': [urgent], 'threshold': [0.65], 'decision': [result]}).to_csv(os.path.join(OUT, 'ticket_decision.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame({'id': selected['case_id'], 'clean': X}).to_csv(os.path.join(OUT, 'usable_tickets.csv'), index=False, encoding='utf-8-sig')
print('参与训练或测试的工单', len(selected), '测试准确率', round(test_accuracy, 4))
print('查询紧急概率、判定', round(urgent, 4), result)

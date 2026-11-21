# B01：参考实现。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "B_客服运营与设备巡检", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "B01")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

tickets = pd.read_csv(os.path.join(DATA, 'tickets.csv'))
# 渠道无大小写要求；na=False 才能稳定地用作掩码。
web = tickets['channel'].str.contains('web', case=False, regex=False, na=False)
selected = tickets.loc[web & tickets['body'].notna()].copy()
punctuation = '，。？！：?!,:'
# 两串长度相同才是一一映射；本题替换为分隔空格，不删除。
translate_table = str.maketrans(punctuation, ' ' * len(punctuation))

def prepare(text):
    keep = []
    # 先换标点再分词，避免让无关标点成为特征。
    for word in jieba.cut(text.translate(translate_table)):
        if word.strip() and word != '请问':
            keep.append(word)
    return ' '.join(keep)

X = selected['body'].map(prepare)
y = selected['priority']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=47, stratify=y)
tfidf = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)
train_vec = tfidf.fit_transform(X_train)
# 测试集不得重建列或重新学习权重。
test_vec = tfidf.transform(X_test)
model = LogisticRegression(max_iter=500)
model.fit(train_vec, y_train)
with open(os.path.join(DATA, 'ticket_query.txt'), encoding='utf-8') as file:
    query = prepare(file.read())
probabilities = model.predict_proba(tfidf.transform([query]))
# 概率列与 classes_ 一一对应；此处靠类别名定位。
urgent_col = list(model.classes_).index('紧急')
urgent = float(probabilities[0, urgent_col])
# 题设边界含等号，不用模型默认的类别阈值代替。
result = '紧急' if urgent >= 0.65 else '普通'
test_accuracy = accuracy_score(y_test, model.predict(test_vec))
pd.DataFrame({'p_urgent': [urgent], 'threshold': [0.65], 'decision': [result]}).to_csv(os.path.join(OUT, 'ticket_decision.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame({'id': selected['case_id'], 'clean': X}).to_csv(os.path.join(OUT, 'usable_tickets.csv'), index=False, encoding='utf-8-sig')
print('参与训练或测试的工单', len(selected), '测试准确率', round(test_accuracy, 4))
print('查询紧急概率、判定', round(urgent, 4), result)

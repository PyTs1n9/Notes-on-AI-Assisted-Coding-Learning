# TF-IDF 文本向量化

[返回示例总目录](../README.md)。每段独立运行，下面是 `BaiQing3.10AI` 的真实输出。

## 函数卡

- **作用**：把一批文档映射成同词列的数值矩阵，常见词在多篇文档中出现时会受到折减。
- **语法**：`TfidfVectorizer(*, input="content", encoding="utf-8", decode_error="strict", strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, analyzer="word", stop_words=None, token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=np.float64, norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False)`；`vectorizer.fit_transform(raw_documents, y=None)`；`vectorizer.transform(raw_documents)`。
- **关键参数**：`tokenizer=str.split` 配合 `token_pattern=None` 接收已经用空格分开的词；`min_df` 可过滤极罕见词；`ngram_range` 设词组长度。
- **返回值**：稀疏矩阵，一行一篇文档，一列一个训练词条；由 `get_feature_names_out()` 查列名。
- **常见坑**：只用训练文档拟合词表。默认分词规则忽略单字；未知词不会创建新列，整条都未知时得到全零向量。
- **适用场景**：分类训练、站内搜索、相似工单推荐。

## 示例 1：看出常用模板词权重

对比工单常见词与只出现一次的关键词，而不是光看词频。

运行 [示例01.py](示例01.py)。

```text
第一条的服务权重 0.5085
第一条的维修权重 0.861
```

## 示例 2：新文本遇到单字和未知词

复用训练词表，区分「命中单字」和「完全没有训练词」。

运行 [示例02.py](示例02.py)。

```text
训练列 ['停', '恢复', '电', '电梯', '门禁']
训练宽度、查询宽度 5 5
两条查询的非零特征数 [1, 0]
```

## 示例 3：加上词组区分相反短语

词完全相同但顺序不同，词组特征能帮忙区分「取消后恢复」与「恢复后取消」。

运行 [示例03.py](示例03.py)。

```text
只有单词时两条相同 True
加入词组后两条相同 False
词组模式列名 ['取消', '取消 恢复', '恢复', '恢复 取消']
```

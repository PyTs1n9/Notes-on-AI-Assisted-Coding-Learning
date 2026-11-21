# 加入设备专有词：让工单中的设备名被当成一个完整词，而不是几个无关片段。

import jieba  # 提供分词及加词方法。

product = '澄光智检终端'  # 模拟的设备专有名。
text = product + '启动失败'  # 一条实际使用场景中的工单。
before = list(jieba.cut(text, HMM=False))  # 暂停生词猜测，先观察默认词表如何切分。
jieba.add_word(product, freq=100000)  # 在当前进程加入这个名称及词频。
after = list(jieba.cut(text, HMM=False))  # 对同一条工单重新分词。
print('添加前', before)  # 观察名称原来如何拆分。
print('添加后', after)  # 检查能否直接按完整设备名筛选。

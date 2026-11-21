# 分隔词时替换为空格而不删除：清理后仍能把相邻的设备词分开。

text = '门禁，电梯；空调'  # 每个短语之间原来有标点。
marks = '，；'  # 只处理这次遇到的两种标点。
as_spaces = str.maketrans(marks, ' ' * len(marks))  # 一一映射为等量空格。
removed = text.translate(str.maketrans('', '', marks))  # 对照：直接删除分隔字符。
separated = text.translate(as_spaces)  # 正式处理：替换后还有分隔位置。
print('删除后按空白切', removed.split())  # 查看为什么直接删不合适。
print('替换后按空白切', separated.split())  # 三个关键词仍相互独立。

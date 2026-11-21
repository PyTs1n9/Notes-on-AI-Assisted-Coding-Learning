# 用字典统一全角文本：把写法不同的编号转成同样格式，甚至把一个特殊符号换成多个字符。

raw = 'ＡＢ：７℃※'  # 模拟设备标签上的全角字符和多余脚注。
mapping = str.maketrans({'Ａ': 'A', 'Ｂ': 'B', '：': ':', '７': '7', '℃': ' degC', '※': None})  # 允许一个字符变多字符或被删除。
normalized = raw.translate(mapping)  # 最终保留统一格式的标签。
print('原标签', raw)  # 原件仍可用于追溯。
print('规范标签', normalized)  # 方便后面的解析或匹配。

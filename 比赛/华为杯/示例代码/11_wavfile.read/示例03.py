# 把 uint8 采样还原成以零为中心的波形：区分无符号 PCM 静音中心和有符号 PCM 的零，避免把直流偏置误当成信号。

import os  # 按当前脚本定位随附数据。
import numpy as np  # 转换成适合负值的类型。
from scipy.io import wavfile  # 读取 WAV 的采样值类型和样本。

# 文件与示例一同移动。
here = os.path.dirname(os.path.abspath(__file__))
# 刻意提供 8 位无符号数据。
path = os.path.join(here, 'data', 'unsigned_pcm8.wav')
# WAV 的 dtype 由实际编码决定。
sr, samples = wavfile.read(path)
# 这段转换只适用于约定的无符号 8 位格式。
if samples.dtype != np.uint8:
	# 避免其他格式误减 128。
	raise ValueError('此示例要求 uint8 PCM')
# 先转类型再居中，避免小数值减法绕回 255。
wave = (samples.astype(np.float64) - 128.0) / 128.0
# 原信号围绕 128 变化。
print('读取类型、采样率、原均值', samples.dtype, sr, float(samples.mean()))
# 能看见正负采样值。
print('居中后的最小值、最大值、均值', float(wave.min()), float(wave.max()), round(float(wave.mean()), 6))

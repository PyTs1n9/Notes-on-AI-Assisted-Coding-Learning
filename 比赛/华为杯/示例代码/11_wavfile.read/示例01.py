# 读取单声道 PCM16 并汇报时长：从文件得到采样率、帧数、时长和有符号峰值，不假定固定采样率。

import os  # 以脚本所在位置找到资源。
import numpy as np  # 计算峰值。
from scipy.io import wavfile  # 读 WAV 头和采样帧。

here = os.path.dirname(os.path.abspath(__file__))  # 工程移动后路径仍可用。
path = os.path.join(here, 'data', 'mono_pcm16.wav')  # 已随示例提供的模拟音频。
sr, samples = wavfile.read(path)  # 顺序为采样率、采样值。
duration = len(samples) / sr  # 长度是帧数；分母是每秒帧数。
floating = samples.astype(np.float64)  # 转换后求绝对值，避免 int16 最小值取绝对值溢出。
peak = np.abs(floating).max()  # 统计 PCM 单位下的最高绝对振幅。
print('采样率、帧数、存储类型', sr, len(samples), samples.dtype)  # 这些值来自文件本身。
print('时长（秒）、绝对峰值', duration, float(peak))  # 不以字节数替代采样帧数。

# 读取单声道 PCM16 并汇报时长：从文件得到采样率、帧数、时长和有符号峰值，不假定固定采样率。

import os  # 以脚本所在位置找到资源。
import numpy as np  # 计算峰值。
from scipy.io import wavfile  # 读 WAV 头和采样帧。

# 工程移动后路径仍可用。
here = os.path.dirname(os.path.abspath(__file__))
# 已随示例提供的模拟音频。
path = os.path.join(here, 'data', 'mono_pcm16.wav')
# 顺序为采样率、采样值。
# sr 每秒钟记录多少个声音采样点，单位是 Hz。例如 44100 Hz 表示每秒记录 44100 个采样点。采样点就是在某个确定时间，测量声音振幅后得到的一个数值。
# PCM 序列：声音波形经过数字化后得到的一串数值。每个数表示某一时刻声音振幅的大小。
sr, samples = wavfile.read(path)
print(samples.shape)
# 长度是帧数；分母是每秒帧数。
duration = len(samples) / sr
# 转换后求绝对值，避免 int16 最小值取绝对值溢出。
floating = samples.astype(np.float64)
# 统计 PCM 单位下的最高绝对振幅。
peak = np.abs(floating).max()
# 这些值来自文件本身。
print('采样率、帧数、存储类型', sr, len(samples), samples.dtype)
# 不以字节数替代采样帧数。
print('时长（秒）、绝对峰值', duration, float(peak))

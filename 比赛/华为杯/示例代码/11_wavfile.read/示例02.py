# 读取立体声，观察混成单声道后的抵消：两个通道中的相反分量会相互抵消；先做形状核对，再按每帧平均。

import os  # 组成文件路径。
import numpy as np  # 计算通道绝对峰值。
from scipy.io import wavfile  # 读取两通道 WAV。

# 使用当前示例的目录。
here = os.path.dirname(os.path.abspath(__file__))
# 其两个声道有一部分反相。
path = os.path.join(here, 'data', 'stereo_pcm16.wav')
# 第 0 维是时间，第 1 维是声道。
sr, stereo = wavfile.read(path)
print(stereo.shape)
# 转成浮点后再做通道运算。
floating = stereo.astype(np.float64)
# 平均同一时刻的两条数据，仍有同样多的时间点。
mono = floating.mean(axis=1)
# 在时间方向汇总，左右声道各得到一个值。
channel_peak = np.abs(floating).max(axis=0)
# 混音后重新统计，而不是把两个峰值直接平均。
mono_peak = np.abs(mono).max()
# 检查没有把时间轴压成两个元素。
print('采样率、转换前后形状', sr, stereo.shape, mono.shape)
# 本例能看到反相声波抵消。
print('各声道峰值、混音峰值', channel_peak.tolist(), float(mono_peak))

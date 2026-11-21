# 从 5 毫秒记录间隔建立频率轴：采集仪给的是时间间隔而不是采样率，检查两边的频率和值是否对应。

import numpy as np  # 将样本下标换算为时间和频率。

step_s = 5 / 1000  # 原始记录表示每 5 毫秒采一帧；转换到秒。
n = 80  # 有 80 条均匀采样。
t = np.arange(n) * step_s  # 样本时间以秒表示。
values = np.cos(2 * np.pi * 25 * t)  # 校准用 25 Hz 余弦。
transformed = np.fft.fft(values)  # 复数分量顺序与 fftfreq 一样。
f = np.fft.fftfreq(n, d=step_s)  # 此处已知的是相邻样本的秒数。
positive = np.flatnonzero(f > 0)  # 忽略零频和负频，只选正频下标。
peak_index = positive[int(np.argmax(np.abs(transformed[positive])))]  # 找回原始 FFT 下标。
print('相邻正频格点差（Hz）', float(f[2] - f[1]))  # 格点间隔取决于点数和采样间隔。
print('峰下标、峰所在频率', int(peak_index), float(f[peak_index]))  # 用真实频率标签解释该下标。

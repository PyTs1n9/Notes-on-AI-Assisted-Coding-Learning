# 偶数点 FFT 的 Nyquist 频点不在正频掩码内：示例输入每帧在正负之间交替；对照全谱与实数单边谱如何记同一个端点。

import numpy as np  # 对比两种配套的频率轴和变换。

sr = 1600  # Nyquist 是这台设备采样率的一半。
signal = np.tile([1.0, -1.0], 8)  # 每个周期跨两个采样点。
n = len(signal)  # 偶数个点才有这个独立端点。
full = np.fft.fft(signal)  # 频点顺序为零、正、负。
full_f = np.fft.fftfreq(n, d=1 / sr)  # 对全谱的端点以负号记频率。
single = np.fft.rfft(signal)  # 对实信号只保留非负频一侧。
single_f = np.fft.rfftfreq(n, d=1 / sr)  # 这套轴以正号记 Nyquist。
full_index = int(np.argmax(np.abs(full)))  # 在全谱里找唯一主要分量。
single_index = int(np.argmax(np.abs(single)))  # 在实数谱里找同一个端点。
print('全频谱、实数谱的最强频点', float(full_f[full_index]), float(single_f[single_index]))  # 同一物理端点的两种标签。
print('全轴严格正频最大值', float(full_f[full_f > 0].max()))  # 用大于零筛选时端点被排除。

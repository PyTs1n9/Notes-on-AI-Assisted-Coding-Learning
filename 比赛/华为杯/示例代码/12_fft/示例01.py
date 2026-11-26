# 从双音信号取出两个主频与幅度：把频域峰和设备中的两个叠加周期关联起来。本例采用单边幅值，不是试卷 A07 的绘图口径。
# 合成信号、做 FFT 并查最大分量。
import numpy as np

# 每秒采样点数。
sr = 2048
# 一秒的样本；选的正弦周期在这一秒内闭合。
n = 2048
# 时间轴以秒为单位。
t = np.arange(n) / sr
# 两个频率的正弦叠加。
sound = 0.65 * np.sin(2 * np.pi * 120 * t) + 0.30 * np.sin(2 * np.pi * 308 * t)
# 返回含正频率、负频率与零频的复数序列。
fft_result = np.fft.fft(sound)
# 为每个复数提供频点标签。
f = np.fft.fftfreq(n, d=1 / sr)
# 排除 0 Hz 和负频率；本例也排除以负值表示的 Nyquist 频点。
positive = f > 0
# 频率轴与幅值使用同一个掩码。
positive_f = f[positive]
# 对本例非端点的单边正弦幅值乘 2。
amplitude = 2 * np.abs(fft_result[positive]) / n
# 只取幅值最大的两个下标，从大到小。
strongest = np.argsort(amplitude)[-2:][::-1]
# 每一个下标都回同一数组取频率与幅度。
for index in strongest:
	# 打印实测峰。
	print('频率（Hz）、幅值', float(positive_f[index]), round(float(amplitude[index]), 3))

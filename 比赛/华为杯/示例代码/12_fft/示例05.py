# 采样频率为 1000 Hz，采样时长为 1 秒。
# 生成两个正弦波叠加的信号，40 Hz 的振幅为 2，100 Hz 的振幅为 0.5。
# 使用 NumPy 计算 FFT，并计算频率数组和振幅数组。
# 用 Matplotlib 画出频谱，横轴显示 0～200 Hz。
import matplotlib.pyplot as plt
import numpy as np

fs = 1000
t = np.arange(0, 1, 1 / fs)
x = 2 * np.sin(2 * np.pi * 40 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
N = len(x)
Y = np.fft.rfft(x)
freq = np.fft.rfftfreq(N, 1 / fs)
amplitude = np.abs(Y) * 2 / N
plt.plot(freq, amplitude)
plt.title('freq-amplitude')
plt.xlim(0, 200)
plt.show()

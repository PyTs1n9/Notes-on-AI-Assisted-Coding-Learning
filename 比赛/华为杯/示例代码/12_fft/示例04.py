import numpy as np
import matplotlib.pyplot as plt

# 采样参数
# 采样频率：1000 Hz
fs = 1000
t = np.arange(0, 1, 1 / fs)

# 原始信号：50 Hz + 120 Hz
x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

# FFT
N = len(x)
freq = np.fft.rfftfreq(N, 1 / fs)
print(freq)
Y = np.fft.rfft(x)
print(Y)
# np.fft.rfft(x) 把时域信号转换为频域数据
amplitude = np.abs(np.fft.rfft(x)) * 2 / N
print(amplitude)
# 绘图
plt.plot(freq, amplitude)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 200)
plt.grid()
plt.show()

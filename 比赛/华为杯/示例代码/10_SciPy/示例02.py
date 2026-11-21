# 用低通保留慢变化、压制高频扰动：相同长度的输入和输出做频率成分对比，验证滤波确实保留低频。

import numpy as np  # 构造已知由慢信号与高频扰动叠加的记录。
from scipy.signal import butter, sosfiltfilt  # 设计双二阶节滤波器并离线前后滤波。

sr = 1000  # 每秒 1000 个采样点。
t = np.arange(sr) / sr  # 相邻采样间隔为 1/sr 秒。
signal = np.sin(2 * np.pi * 35 * t)  # 需要保留的慢变化成分。
noise = 0.8 * np.sin(2 * np.pi * 230 * t)  # 模拟高频扰动。
mixed = signal + noise  # 对混合信号滤波，不去访问纯净信号做预测。
sos = butter(4, 80, btype='lowpass', fs=sr, output='sos')  # 截止频率 80 Hz；明示单位。
filtered = sosfiltfilt(sos, mixed)  # 适用于完整离线信号。
before = np.abs(np.fft.rfft(mixed))  # 比较滤波前的正频率幅值。
after = np.abs(np.fft.rfft(filtered))  # 同长度、同频率坐标下比较滤波后的幅值。
freq = np.fft.rfftfreq(len(mixed), d=1 / sr)  # 根据采样间隔建立 Hz 轴。
for hz in [35, 230]:  # 比较目标信号和扰动，不能只看图是否变平。
    i = int(np.argmin(np.abs(freq - hz)))  # 找目标 Hz 的频率下标。
    print('频率、滤波前后幅值', hz, round(float(before[i]), 2), round(float(after[i]), 2))  # 使用相同尺度作相对比较。

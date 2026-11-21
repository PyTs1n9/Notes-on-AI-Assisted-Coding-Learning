# 不同采样率下，同一音对应不同下标：两个设备采相同的音，固定取相同行数；不要用 FFT 下标判断两声是否相同。

import numpy as np  # 创建两组样本，按各自采样率算频率。

n = 500  # 两台设备各提供相同数目的采样点。
peaks = []  # 保存实际频点而非只记录索引。
indices = []  # 同时记录索引作为对照。
for sr in [2000, 3000]:  # 两台模拟设备的采样率不同。
    t = np.arange(n) / sr  # 时间跨度会随采样率改变。
    values = np.sin(2 * np.pi * 120 * t)  # 模拟同一个物理频率。
    mag = np.abs(np.fft.fft(values))  # 返回双边频谱的幅值。
    frequencies = np.fft.fftfreq(n, d=1 / sr)  # 必须用该设备自己的采样间隔。
    pos = np.flatnonzero(frequencies > 0)  # 只在正频部分找峰。
    index = int(pos[np.argmax(mag[pos])])  # 映射回完整频率轴下标。
    indices.append(index)  # 下标可能不同。
    peaks.append(float(frequencies[index]))  # 比较的是以 Hz 标记的值。
    print('采样率、峰下标、峰频率', sr, index, peaks[-1])  # 一行包含两种读法。

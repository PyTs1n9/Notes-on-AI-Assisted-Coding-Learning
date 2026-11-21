# 短音频补零后比较频率格点：同一段输入不追加真实采样；观察稀疏格点与更密格点上的峰位置。

import numpy as np  # 生成窗与频率谱。

sr = 8000  # 固定采样率，不能用补零点数代替它。
t = np.arange(800) / sr  # 记录时长保持不变。
tone = np.sin(2 * np.pi * 443.3 * t)  # 真实频率不刚好落在粗格点上。
tapered = tone * np.hanning(len(tone))  # 同一窗用于两种 FFT 点数，保持比较公平。
estimates = []  # 存储两种估计，便于比较。
for n in [len(tapered), 8192]:  # 第二次在末端补零，真实输入完全一样。
    f = np.fft.fftfreq(n, d=1 / sr)  # 频率轴与此次点数保持一致。
    amp = np.abs(np.fft.fft(tapered, n=n))  # 对同一数据使用本次点数变换。
    mask = f > 0  # 只看正频率部分。
    peak = float(f[mask][np.argmax(amp[mask])])  # 峰的位置只能在已有的格点里挑。
    estimates.append(peak)  # 比较同一音频用不同格点的估计。
    print('点数、相邻格点差、峰位置', n, round(float(f[1] - f[0]), 4), round(peak, 4))  # 频率单位都是 Hz。
print('两次的真实采样时长（秒）', t.size / sr)  # 补零没有录到更多真实声音。

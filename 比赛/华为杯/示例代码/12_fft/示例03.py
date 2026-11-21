# 对称删掉一对干扰频点再还原：对可控的整数周期信号，移除窄频干扰时同时处理正负频率，再变回时域。

import numpy as np  # 用正、反 FFT 观察一个可验证的离线简例。

sr = 2048  # 采样率。
t = np.arange(sr) / sr  # 一整秒。
useful = np.sin(2 * np.pi * 136 * t)  # 需要保留的成分。
buzz = 0.45 * np.sin(2 * np.pi * 720 * t)  # 演示要消掉的干扰。
observed = useful + buzz  # 模拟实际收到的信号。
spectrum = np.fft.fft(observed)  # 得到带正负频率的复数序列。
f = np.fft.fftfreq(len(observed), d=1 / sr)  # 查每个分量对应的符号和频率。
corrupt = np.isclose(np.abs(f), 720)  # 同时磨掉 +720 Hz 和 -720 Hz，保持实信号所需的对称性。
cleaned = spectrum.copy()  # 保留原频谱便于比较。
cleaned[corrupt] = 0  # 此处的合成信号恰好无泄漏，不泛化为所有真实音频的滤波做法。
restored = np.fft.ifft(cleaned)  # 回到时域；浮点误差会产生极小的虚部。
print('删除的频点（Hz）', f[corrupt].tolist())  # 验证确实删的是一对频点。
print('恢复信号虚部最大绝对值', float(np.abs(restored.imag).max()))  # 若只删一边，虚部不会接近零。

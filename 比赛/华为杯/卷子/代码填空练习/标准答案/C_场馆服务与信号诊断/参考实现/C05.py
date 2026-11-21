# C05：参考实现。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "C_场馆服务与信号诊断", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "C05")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile

sr, raw = wavfile.read(os.path.join(DATA, 'switch.wav'))
audio = raw.astype(np.float64)
window = int(sr * 0.5)
n_fft = window * 4
# 补零后频率点数必须与 FFT 相同；点更密不等于录得了更长信号。
freq = np.fft.fftfreq(n_fft, d=1 / sr)
band = (freq >= 200) & (freq <= 900)
starts = []
peaks = []
for start in range(0, len(audio) - window + 1, window):
    # 这里先截一个完整时间窗，不是每次从开头累计。
    part = audio[start:start + window]
    tapered = (part - part.mean()) * np.hanning(window)
    # 指定更长变换点数才会补零；不改变原时间窗。
    spectrum = np.fft.fft(tapered, n=n_fft)
    # 同一个频带掩码同时用于频率轴和频谱，才能正确找峰。
    j = np.argmax(np.abs(spectrum[band]))
    peaks.append(float(freq[band][j]))
    starts.append(start / sr)
# 在相邻峰值之间求一阶差分，输出比窗数少 1。
jumps = np.abs(np.diff(peaks))
found = np.flatnonzero(jumps >= 100)
change_at = None
if found.size:
    # 差分下标 0 对应第 1 和第 2 窗之间，所以切换时刻用后一窗起点。
    change_at = starts[int(found[0]) + 1]
result = pd.DataFrame({'start_s': starts, 'peak_hz': peaks})
result['delta_hz'] = result['peak_hz'].diff().abs()
result.to_csv(os.path.join(OUT, 'segment_peaks.csv'), index=False)
pd.DataFrame({'changed': [change_at is not None], 'time_s': [change_at]}).to_csv(os.path.join(OUT, 'first_change.csv'), index=False)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(starts, peaks, marker='o')
if change_at is not None:
    ax.axvline(change_at, color='red', linestyle='--')
ax.set(xlabel='Window start (s)', ylabel='Peak frequency (Hz)')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'frequency_transition.png'), dpi=120)
plt.close(fig)
print('各窗主频', peaks)
if change_at is None:
    print('首次切换 no change')
else:
    print('首次切换（秒）', change_at)

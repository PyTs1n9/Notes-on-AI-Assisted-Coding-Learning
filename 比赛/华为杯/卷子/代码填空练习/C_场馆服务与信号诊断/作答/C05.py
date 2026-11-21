# C05：只替换带题号的空位。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))  # 随卷数据目录。
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
freq = np.fft.fftfreq(__C05_01__)
band = (freq >= 200) & (freq <= 900)
starts = []
peaks = []
for start in range(0, len(audio) - window + 1, window):
    part = audio[__C05_02__]
    tapered = (part - part.mean()) * np.hanning(window)
    spectrum = np.fft.fft(__C05_03__)
    j = np.argmax(np.abs(__C05_04__))
    peaks.append(float(freq[band][j]))
    starts.append(start / sr)
jumps = np.abs(__C05_05__)
found = np.flatnonzero(jumps >= 100)
change_at = None
if found.size:
    change_at = starts[__C05_06__]
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

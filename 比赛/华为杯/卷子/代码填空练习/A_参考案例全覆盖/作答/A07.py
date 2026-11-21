# A07：只替换带题号的空位。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "A07")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile

n_fft = 4096
rows = []
fig, axes = plt.subplots(2, 2, figsize=(11, 6))
for index, name in enumerate(['1.wav', '2.wav']):
    sr, raw = __A07_01__
    audio = raw.astype(np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=__A07_02__)
    duration = __A07_03__
    fft_result = np.fft.__A07_04__(audio, n=n_fft)
    freq_axis = np.fft.__A07_05__(n=n_fft, d=1 / sr)[:n_fft // 2]
    mag = np.abs(fft_result)[:n_fft // 2] / __A07_06__
    peak_i = int(np.argmax(__A07_07__)) + 1
    rows.append({'name': name, 'sr': sr, 'duration': duration, 'mean': np.mean(audio),
                 'median': np.median(audio), 'std': np.std(audio), 'abs_peak': np.max(np.abs(audio)),
                 'peak_hz': float(freq_axis[peak_i])})
    axes[index, 0].plot(np.arange(len(audio)) / sr, audio, linewidth=0.4)
    axes[index, 0].set(xlabel='Time (s)', ylabel='Sample value', title=name)
    axes[index, 1].plot(freq_axis, mag)
    axes[index, 1].set(xlabel='Frequency (Hz)', ylabel='Scaled magnitude', title=name)
first, second = rows
relative = __A07_08__
hz_gap = abs(second['peak_hz'] - first['peak_hz'])
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'audio_analysis.png'), dpi=120)
plt.close(fig)
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'audio_stats.csv'), index=False)
pd.DataFrame({'relative_amplitude_difference': [relative], 'absolute_peak_hz_difference': [hz_gap]}).to_csv(os.path.join(OUT, 'comparison.csv'), index=False)
print('采样率、时长、峰值频率', [(row['sr'], row['duration'], row['peak_hz']) for row in rows])
print('绝对峰值相对差、主频绝对差', round(float(relative), 4), hz_gap)

# B05：只替换带题号的空位。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "B05")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile

records = []
fig, axes = plt.subplots(2, 1, figsize=(8, 6))
for index, name in enumerate(['idle.wav', 'running.wav']):
    sr, samples = wavfile.read(os.path.join(DATA, name))
    audio = samples.astype(np.float64)
    centered = __B05_01__
    n = len(centered)
    spectrum = np.fft.fft(__B05_02__)
    frequencies = np.fft.fftfreq(__B05_03__)
    positive = frequencies > 0
    f = frequencies[positive]
    mag = np.abs(spectrum[positive]) / n
    in_band = __B05_04__
    band_f = f[in_band]
    band_mag = mag[in_band]
    peak_band = float(band_f[__B05_05__])
    peak_global = float(f[np.argmax(mag)])
    records.append({'source': name, 'global_hz': peak_global, 'band_hz': peak_band})
    axes[index].plot(f, mag)
    axes[index].axvspan(250, 600, alpha=0.2, color='orange')
    axes[index].axvline(peak_band, color='red')
    axes[index].set(xlim=(0, 900), xlabel='Frequency (Hz)', ylabel='Magnitude', title=name)
delta = abs(records[1]['band_hz'] - records[0]['band_hz'])
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'band_spectra.png'), dpi=120)
plt.close(fig)
pd.DataFrame(records).to_csv(os.path.join(OUT, 'band_peaks.csv'), index=False)
pd.DataFrame({'comparison': ['absolute_band_peak_difference'], 'hz': [delta]}).to_csv(os.path.join(OUT, 'band_difference.csv'), index=False)
print('各段全谱峰、带内峰', [(row['global_hz'], row['band_hz']) for row in records])
print('带内峰绝对差', delta)

# C04：只替换带题号的空位。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "C04")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rgb = mpimg.imread(os.path.join(DATA, 'hall.jpg'))
ranges = {'A': (12, 60, 18, 75), 'B': (71, 129, 102, 203)}
stats = []
hist_record = {}
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
for index, (name, coordinates) in enumerate(ranges.items()):
    y0, y1, x0, x1 = coordinates
    roi = rgb[__C04_01__]
    counts, edges = np.histogram(roi[:, :, 0], bins=8, range=(0, 256))
    weights = __C04_02__
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    amp = __C04_03__
    strong_fraction = float(__C04_04__)
    stats.append({'region': name, 'pixels': gray.size, 'strong_edge_fraction': strong_fraction})
    hist_record[name] = weights
    axes[index].imshow(roi)
    axes[index].set_title('Region ' + name)
    axes[index].axis('off')
    axes[2].stairs(weights, edges, label=name)
axes[2].set(xlabel='Red intensity', ylabel='Fraction per bin')
axes[2].legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'region_features.png'), dpi=120)
plt.close(fig)
pd.DataFrame(stats).to_csv(os.path.join(OUT, 'region_stats.csv'), index=False)
pd.DataFrame({'left_edge': edges[:-1], 'A': hist_record['A'], 'B': hist_record['B']}).to_csv(os.path.join(OUT, 'region_histograms.csv'), index=False)
print('两区域像素数', [row['pixels'] for row in stats])
print('两区域强边缘占比', [round(row['strong_edge_fraction'], 4) for row in stats])

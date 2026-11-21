# B04：只替换带题号的空位。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "B04")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rgb = mpimg.imread(os.path.join(DATA, 'inspection.jpg'))
gray = cv2.cvtColor(rgb, __B04_01__)
gx = cv2.Sobel(gray, cv2.CV_32F, __B04_02__, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_32F, __B04_03__, ksize=3)
mag = cv2.magnitude(gx, gy)
strong = __B04_04__
x_edge = strong & __B04_05__
y_edge = strong & __B04_06__
limits = [0, 32, 96, 160, 208, 256]
counts, edges = np.histogram(__B04_07__)
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
axes[0].imshow(rgb)
axes[0].axis('off')
axes[1].imshow(x_edge, cmap='gray')
axes[1].set_title('X-dominant strong edges')
axes[1].axis('off')
axes[2].imshow(y_edge, cmap='gray')
axes[2].set_title('Y-dominant strong edges')
axes[2].axis('off')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'edge_directions.png'), dpi=120)
plt.close(fig)
pd.DataFrame({'kind': ['strong', 'x', 'y'], 'count': [int(strong.sum()), int(x_edge.sum()), int(y_edge.sum())]}).to_csv(os.path.join(OUT, 'edge_counts.csv'), index=False)
pd.DataFrame({'left': edges[:-1], 'right': edges[1:], 'count': counts}).to_csv(os.path.join(OUT, 'green_histogram.csv'), index=False)
print('强边缘、x 占优、y 占优', int(strong.sum()), int(x_edge.sum()), int(y_edge.sum()))

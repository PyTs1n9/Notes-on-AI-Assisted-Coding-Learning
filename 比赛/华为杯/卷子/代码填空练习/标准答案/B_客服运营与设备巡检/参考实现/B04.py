# B04：参考实现。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "B_客服运营与设备巡检", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "B04")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rgb = mpimg.imread(os.path.join(DATA, 'inspection.jpg'))
# 输入通道顺序不同会改变灰度和最终边缘。
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
# 对 x 方向求一阶导；x 表示亮度变化方向，不是边线朝向。
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
# 对 y 方向求一阶导；不要把两个参数对写成一样。
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
mag = cv2.magnitude(gx, gy)
# 使用合成幅值筛选强边缘，阈值边界包含等号。
strong = mag >= 120
# 方向占优比较绝对值，否则负的强梯度会被误判。
x_edge = strong & (np.abs(gx) >= np.abs(gy))
# 剩余强边缘全部归 y，不能把弱边缘也选进去。
y_edge = strong & ~x_edge
limits = [0, 32, 96, 160, 208, 256]
# 绿色位于 RGB 的下标 1；边界列表允许箱宽不等。
counts, edges = np.histogram(rgb[:, :, 1], bins=limits)
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

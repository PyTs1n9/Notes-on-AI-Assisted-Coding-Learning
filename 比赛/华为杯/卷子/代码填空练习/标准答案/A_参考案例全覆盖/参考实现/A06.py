# A06：参考实现。
# 通过脚本位置找文件，PyCharm 的工作目录不同也能运行。
import os  # 处理路径与输出目录。
HERE = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录。
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "A_参考案例全覆盖", "data"))  # 随卷数据目录。
OUT = os.path.join(HERE, "outputs", "A06")  # 本题产物目录；不覆盖输入。
os.makedirs(OUT, exist_ok=True)  # 第一次运行也能保存。

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rgb = mpimg.imread(os.path.join(DATA, 'airplane.jpg'))
height, width, channels = rgb.shape
hist = {'bin_left': np.arange(0, 256, 16)}
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(rgb)
axes[0].set_title('Original RGB')
axes[0].axis('off')
for channel, color in enumerate(['red', 'green', 'blue']):
    # 固定最后一维的通道索引，不能把三个通道混成一份分布。
    counts, edges = np.histogram(rgb[:, :, channel], bins=16, range=(0, 256))
    hist[color] = counts
    axes[1].stairs(counts, edges, color=color, label=color)
axes[1].legend()
axes[1].set(xlabel='Intensity', ylabel='Pixels')
# Matplotlib 读出的这张 JPG 为 RGB，用 RGB 的灰度转换常量。
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
# x 方向一阶导检测横向亮度变化，浮点深度保存负梯度。
grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
# y 方向一阶导检测纵向亮度变化；参数顺序与 x 方向相反。
grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
# magnitude 合成两方向的欧氏幅值，不是任一个方向的绝对值。
amp = cv2.magnitude(grad_x, grad_y)
axes[2].imshow(amp, cmap='magma')
axes[2].set_title('Gradient magnitude')
axes[2].axis('off')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'image_features.png'), dpi=120)
plt.close(fig)
pd.DataFrame(hist).to_csv(os.path.join(OUT, 'rgb_histogram.csv'), index=False)
pd.DataFrame({'height': [height], 'width': [width], 'channels': [channels],
              'gradient_mean': [amp.mean()], 'gradient_max': [amp.max()]}).to_csv(os.path.join(OUT, 'image_stats.csv'), index=False)
print('高 宽 通道数', height, width, channels)
print('各通道计数', int(hist['red'].sum()), int(hist['green'].sum()), int(hist['blue'].sum()))
print('梯度均值 最大值', round(float(amp.mean()), 3), round(float(amp.max()), 3))

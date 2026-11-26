# 定位竖条的左右边：竖条两侧的亮度变化沿 x，观察 x 与 y 梯度的差异。

import cv2  # 计算 Sobel 方向梯度。
import numpy as np  # 创建可控的灰度图。

# 宽和高不同，便于跟踪轴。
gray = np.zeros((36, 52), dtype=np.uint8)

# 在黑底上放一条贯穿上下的亮色竖条。
gray[:, 19:35] = 190
# 沿列坐标变化求导。
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
# 沿行坐标变化求导。
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

# 取中间一行穿过竖条左右两侧。
row = gx[gray.shape[0] // 2]
# 看竖边在什么列出现。
print('x 方向有梯度的列', np.flatnonzero(row).tolist())
# 竖边主要在 x 导数中显著。
print('x 与 y 最大绝对梯度', float(np.abs(gx).max()), float(np.abs(gy).max()))

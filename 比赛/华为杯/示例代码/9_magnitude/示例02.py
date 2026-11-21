# 找出只看横向会漏掉的边：图上有横向接缝，只有把两个方向的 Sobel 合成幅值才能稳定发现。

import cv2  # 求两方向梯度与幅值。
import numpy as np  # 构造接缝图像。

plate = np.full((28, 46), 38, dtype=np.uint8)  # 先填较暗的灰度值。
plate[14:, :] = 150  # 下面较亮，形成一条水平接缝。
gx = cv2.Sobel(plate, cv2.CV_32F, 1, 0)  # 对水平接缝，沿 x 的变化很小。
gy = cv2.Sobel(plate, cv2.CV_32F, 0, 1)  # 垂直穿过接缝时亮度会跃变。
mag = cv2.magnitude(gx, gy)  # 不预先猜边缘方向，合成强度。
print('只看 x 得到的强边缘', int((np.abs(gx) >= 160).sum()))  # 对照只看一个方向的漏检。
print('看幅值时的强边缘', int((mag >= 160).sum()))  # 统计双向合成的强边缘。

# 用各区域占比比较表面纹理：不同大小区域的边缘点数不可直接比较，使用超过同一阈值的像素占比。

import cv2  # 从灰度图提取梯度并合成。
import numpy as np  # 创建两个大小不一的合成区域。

smooth = np.full((25, 41), 90, dtype=np.uint8)  # 第一块没有纹理。
texture = np.full((42, 34), 90, dtype=np.uint8)  # 第二块面积不同。
for start in [4, 12, 20, 28]:  # 让第二块的部分列形成亮纹。
    texture[:, start:start + 3] = 210  # 有上升边缘和下降边缘。
fractions = []  # 保存各自区域内的强边缘占比。
for image in [smooth, texture]:  # 两组使用相同 Sobel 参数和阈值。
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)  # x 方向梯度。
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)  # y 方向梯度。
    amplitude = cv2.magnitude(gx, gy)  # 对每个像素取双向梯度的长度。
    fraction = float((amplitude >= 120).mean())  # 真值均值就是该区域中强边缘像素的比例。
    fractions.append(fraction)  # 每一项各用自己的区域为分母。
    print('区域像素数、强边缘占比', image.size, round(fraction, 4))  # 区分分母与输出指标。

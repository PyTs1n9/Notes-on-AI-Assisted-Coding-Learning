# 平滑后再查找带噪图像的边：用同一张带噪合成板比较：平滑应降低背景误响应，同时真边缘仍明显。

import cv2  # 提供高斯平滑和 Sobel。
import numpy as np  # 创建固定随机噪声。

# 每次生成同一份噪声用于公平对照。
rng = np.random.default_rng(61)

# 先构造暗左半区。
ideal = np.full((60, 90), 45, dtype=np.float32)
# 右半区较亮，边界在中间。
ideal[:, 45:] = 190

# 把随机波动加到同一输入上。
noisy = ideal + rng.normal(0, 9, ideal.shape).astype(np.float32)
# 小范围加权平均，减弱噪声。
smooth = cv2.GaussianBlur(noisy, (5, 5), 0)
# 原图 x 导数。
grad_before = cv2.Sobel(noisy, cv2.CV_32F, 1, 0)
# 仅比较加平滑后 x 导数。
grad_after = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
# 避开真实接缝计算背景噪声。
background_before = np.abs(grad_before[:, :38]).mean()
# 保持统计同一块背景。
background_after = np.abs(grad_after[:, :38]).mean()
# 观察平滑后真边缘响应。
seam = np.abs(grad_after[:, 44:46]).mean()
# 看误响应是否下降。
print('背景平均梯度：平滑前、后', round(float(background_before), 2), round(float(background_after), 2))
# 真边缘不应消失。
print('平滑后的接缝平均梯度', round(float(seam), 2))

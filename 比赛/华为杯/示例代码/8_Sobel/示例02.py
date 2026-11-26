# 保留暗变亮和亮变暗：灰度图有两个过渡；8 位输出只适合不会丢失预期信息的情况，这里和浮点输出对照。

import cv2  # 提供可指定输出深度的 Sobel。
import numpy as np  # 构造多级灰度带。

# 原始强度在合法的 8 位区间内。
gray = np.full((30, 54), 90, dtype=np.uint8)
# 从较暗左侧进入较亮的带。
gray[:, 12:29] = 230
# 从亮带进入更暗区域。
gray[:, 29:] = 20

# 保持有符号大范围梯度。
floating = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
# 对照：无符号输出会把负数截到 0。
unsigned = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)

# 所有沿 x 从亮到暗的位置。
falling = floating < 0

# 正负两种方向都保留。
print('浮点梯度最小/最大', float(floating.min()), float(floating.max()))
# 查看丢失的负向响应。
print('负梯度位置在无符号结果里的最大值', int(unsigned[falling].max()))

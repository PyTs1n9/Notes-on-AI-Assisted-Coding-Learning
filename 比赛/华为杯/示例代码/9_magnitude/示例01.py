# 找出跑带上的二维移动异常：单个方向的速度不显眼，两个方向一起算时可以超过告警阈值。

import cv2  # 合成两个速度分量。
import numpy as np  # 使用浮点速度阵列。

vx = np.array([[0.0, 2.0, -3.7], [4.2, -1.0, 0.0]], dtype=np.float32)  # 样点的 x 速度。
vy = np.array([[2.0, 0.5, 3.7], [0.0, 2.0, 0.0]], dtype=np.float32)  # 相同样点的 y 速度。
speed = cv2.magnitude(vx, vy)  # 对应位置逐点合成，方向的正负不会抵消。
alert = speed >= 4.5  # 按合成速度判断是否报告。
print('各点合速度', speed.round(2).tolist())  # 查看每个位置的速度大小。
print('超限位置', np.argwhere(alert).tolist())  # 输出行与列位置。

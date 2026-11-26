import cv2
import numpy as np

# 1. 黑底上画一个亮色矩形
gray = np.zeros((160, 220), dtype=np.uint8)
gray[40:120, 60:160] = 190

# 2. 适当模糊，减小细小亮度波动的影响
blur = cv2.GaussianBlur(gray, (3, 3), 0)

# 3. 计算两个方向的梯度，保留原始浮点数结果
gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)

# 4. 合成每个像素的边缘强度
magnitude = cv2.magnitude(gx, gy)

# 5. 按梯度强度筛选：大于 100 的位置显示为白色
edges = (magnitude > 100).astype(np.uint8) * 255

# 6. 把强度图缩放到 0～255，方便显示
magnitude_show = cv2.normalize(
    magnitude, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# 7. 显示结果
cv2.imshow("Original", gray)
cv2.imshow("Sobel X", cv2.convertScaleAbs(gx))
cv2.imshow("Sobel Y", cv2.convertScaleAbs(gy))
cv2.imshow("Magnitude", magnitude_show)
cv2.imshow("Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np

# 1. 创建黑色图片：100 行 × 100 列
img = np.zeros((100, 100), dtype=np.uint8)

# 2. 中间填一个白色方块
img[25:75, 25:75] = 255

# 3. 分别计算两个方向的亮度变化
gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# 4. 取绝对值并转成可显示的 8 位图像
gx = cv2.convertScaleAbs(gx)
gy = cv2.convertScaleAbs(gy)

# 5. 显示结果
cv2.imshow("Original", img)
cv2.imshow("Sobel X", gx)
cv2.imshow("Sobel Y", gy)

cv2.waitKey(0)  # 按任意键关闭窗口
cv2.destroyAllWindows()
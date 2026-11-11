import os
import cv2

input_path = os.path.join('input', 'practice_rgb.png')
os.makedirs('output_practice', exist_ok=True)
output_path = 'output_practice'

image_bgr = cv2.imread(input_path)
print(image_bgr.shape)
print(image_bgr.dtype)
print(image_bgr[0, 0].tolist())

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
print(image_rgb[0, 0].tolist())
origin_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
print(origin_img[0, 0].tolist())

resized = cv2.resize(origin_img, (180, 108), interpolation=cv2.INTER_AREA)
print(resized.shape)
print(resized.dtype)

color = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
print(color.shape)
print(color.dtype)

gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
print(gray.shape)
print(gray.dtype)
print(gray[0, 0].tolist())

horizontal = cv2.flip(resized, 1)
print(horizontal.shape)
print(horizontal.dtype)
vertical = cv2.flip(resized, 0)
print(vertical.shape)
print(vertical.dtype)

copy_path = os.path.join(output_path, 'sample_copy.png')  # 拼出原图副本的输出路径
rgb_path = os.path.join(output_path, 'rgb_restored.png')  # 拼出 RGB 转回 BGR 后的输出路径
resize_path = os.path.join(output_path, 'resized.png')  # 拼出缩放结果的输出路径
gray_path = os.path.join(output_path, 'gray.png')  # 拼出灰度结果的输出路径
horizontal_path = os.path.join(output_path, 'horizontal.png')  # 拼出水平翻转结果的输出路径
vertical_path = os.path.join(output_path, 'vertical.png')  # 拼出垂直翻转结果的输出路径
print('保存 RGB 往返图：', cv2.imwrite(rgb_path, origin_img))  # 函数②：写入转回 BGR 的彩色数组
print('保存缩放图：', cv2.imwrite(resize_path, resized))  # 函数②：写入缩放后的图片
print('保存灰度图：', cv2.imwrite(gray_path, gray))  # 函数②：写入二维灰度数组
print('保存水平翻转图：', cv2.imwrite(horizontal_path, horizontal))  # 函数②：写入左右镜像结果
print('保存垂直翻转图：', cv2.imwrite(vertical_path, vertical))  # 函数②：写入上下镜像结果

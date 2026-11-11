import os
import cv2

INPUT_PATH = os.path.join('input', 'practice_rgb.png')
OUTPUT_DIR = 'output_practice'

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_bgr = cv2.imread(INPUT_PATH, cv2.IMREAD_COLOR)
if image_bgr is None:
    raise FileNotFoundError('无法读取 input/practice_rgb.png')
print('原图 shape：', image_bgr.shape)
print('原图 dtype：', image_bgr.dtype)
print('原图左上角 BGR：', image_bgr[0, 0].tolist())

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
print('左上角 RGB：', image_rgb[0, 0].tolist())
color_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

resized_bgr = cv2.resize(image_bgr, (180, 108), interpolation=cv2.INTER_AREA)
print('缩放图 shape：', resized_bgr.shape)
print('缩放图 dtype：', resized_bgr.dtype)

gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
print('灰度图 shape：', gray.shape)
print('灰度图 dtype：', gray.dtype)
print('灰度图左上角：', gray[0, 0].tolist())

horizontal = cv2.flip(resized_bgr, 1)
vertical = cv2.flip(resized_bgr, 0)
print('水平翻转 shape：', horizontal.shape)
print('垂直翻转 shape：', vertical.shape)

color_path = os.path.join(OUTPUT_DIR, 'color.png')
resize_path = os.path.join(OUTPUT_DIR, 'resized.png')
gray_path = os.path.join(OUTPUT_DIR, 'gray.png')
horizontal_path = os.path.join(OUTPUT_DIR, 'horizontal.png')
vertical_path = os.path.join(OUTPUT_DIR, 'vertical.png')

print('保存彩色图：', cv2.imwrite(color_path, color_bgr))
print('保存缩放图：', cv2.imwrite(resize_path, resized_bgr))
print('保存灰度图：', cv2.imwrite(gray_path, gray))
print('保存水平翻转图：', cv2.imwrite(horizontal_path, horizontal))
print('保存垂直翻转图：', cv2.imwrite(vertical_path, vertical))

for file_name in ['color.png', 'resized.png', 'gray.png', 'horizontal.png', 'vertical.png']:
    file_path = os.path.join(OUTPUT_DIR, file_name)
    saved = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if saved is None:
        raise RuntimeError('读回失败：' + file_name)
    print('读回：', file_name, saved.shape, saved.dtype)

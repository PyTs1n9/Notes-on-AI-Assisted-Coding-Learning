import os, cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE, 'input')
INPUT_PATH = os.path.join(INPUT_DIR, 'practice_aug.png')
os.makedirs('output_practice', exist_ok=True)
OUTPUT_DIR = os.path.join(BASE, 'output_practice')
EXTENSIONS = ('.png')

# img = cv2.imread(INPUT_PATH, cv2.IMREAD_COLOR)
img_decode = cv2.imdecode(np.fromfile(INPUT_PATH, dtype=np.uint8), cv2.IMREAD_COLOR)
if img_decode is None:
	raise FileNotFoundError('File is not found!')
print(img_decode.shape, img_decode.dtype)

rng = np.random.default_rng(37)
angle = float(rng.uniform(-28, 28))
alpha = float(rng.uniform(0.85, 1.35))
beta = int(rng.integers(-35, 35))
print('random angle alpha number :', round(angle, 6), round(alpha, 6), beta)

h, w = img_decode.shape[:2]
center = (w / 2, h / 2)
matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
rotated = cv2.warpAffine(img_decode, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0, 0, 0))

raw = rotated.astype(np.float32) * alpha + beta
print('截断前最小，最大：', float(raw.min()), float(raw.max()))
clipped = np.clip(raw, 0, 255)
bright = clipped.astype(np.uint8)
print('截断后最小，最大：', float(bright.min()), float(bright.max()))

crop_width = 180
crop_height = 120
x = int(rng.integers(0, w - crop_width + 1))
y = int(rng.integers(0, h - crop_height + 1))
crop = bright[y:y + crop_height, x:x + crop_width]

resized = cv2.resize(crop, (144, 96), interpolation=cv2.INTER_AREA)
blurred = cv2.GaussianBlur(resized, (5, 5), sigmaX=1.1, sigmaY=1.1, borderType=cv2.BORDER_REFLECT_101)

file_names = ['rotated.png', 'brightness.png', 'crop.png', 'resized.png', 'blurred.png']  # 按处理顺序列出要保存的文件名
images = [rotated, bright, crop, resized, blurred]  # 同一顺序列出各阶段数组

for index in range(len(file_names)):
	save_path = os.path.join(OUTPUT_DIR, file_names[index])
	ok, buf = cv2.imencode('.png', images[index])
	if ok:
		buf.tofile(save_path)

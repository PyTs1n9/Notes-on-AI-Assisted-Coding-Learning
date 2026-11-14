import os, cv2

import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE, 'input')
INPUT_PATH = os.path.join(INPUT_DIR, 'compare_practice.png')
OUTPUT_DIR = os.path.join(BASE, 'output_demo')

# A
pil_img = Image.open(INPUT_PATH).convert('RGB')
flip_tv = F.hflip(pil_img)
rotate_tv = F.rotate(flip_tv, 0)
tv_img = np.array(rotate_tv)
bgr = cv2.imdecode(np.fromfile(INPUT_PATH, np.uint8), cv2.IMREAD_COLOR)
if bgr is None:
	raise FileNotFoundError('no file')
flip_cv = cv2.flip(bgr, 1)
# opencv旋转角度0是旋转90度
# rotate_cv = cv2.rotate(flip_cv, 0)
h, w = flip_cv.shape[:2]
center = (w / 2, h / 2)
matrix = cv2.getRotationMatrix2D(center, 0, 1.0)
rotate_cv = cv2.warpAffine(flip_cv, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(0, 0, 0))
cv_img = cv2.cvtColor(rotate_cv, cv2.COLOR_BGR2RGB)
print('A未统一颜色比较：', np.array_equal(tv_img, rotate_cv))
print('A统一颜色后比较：', np.array_equal(tv_img, cv_img))

# B
pil_img = Image.open(INPUT_PATH).convert('RGB')
rotate_tv = F.rotate(pil_img, 0)
flip_tv = F.hflip(rotate_tv)
tv_img = np.array(flip_tv)
bgr = cv2.imdecode(np.fromfile(INPUT_PATH, np.uint8), cv2.IMREAD_COLOR)
if bgr is None:
	raise FileNotFoundError('no file')
h, w = bgr.shape[:2]
center = (w / 2, h / 2)
matrix = cv2.getRotationMatrix2D(center, 0, 1.0)
rotate_cv = cv2.warpAffine(bgr, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(0, 0, 0))
flip_cv = cv2.flip(rotate_cv, 1)
cv_img = cv2.cvtColor(flip_cv, cv2.COLOR_BGR2RGB)
print('B未统一颜色比较：', np.array_equal(tv_img, flip_cv))
print('B统一颜色后比较：', np.array_equal(tv_img, cv_img))

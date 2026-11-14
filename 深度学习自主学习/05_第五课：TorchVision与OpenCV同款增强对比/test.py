import os, cv2

import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE, 'input')
INPUT_PATH = os.path.join(INPUT_DIR, 'compare_practice.png')
# 老师删除：原 OUTPUT_DIR 从未使用；本题无需保存图片，删掉无效路径配置。

# 老师新增：原提交缺少运行前预测。以下是教师示范假设，不冒充你的原始预测。
# 假设：零角度只剩下翻转，两种顺序应一致；非零角度可能改变结果，需要实际比较。
pil_img = Image.open(INPUT_PATH).convert('RGB')
bgr = cv2.imdecode(np.fromfile(INPUT_PATH, np.uint8), cv2.IMREAD_COLOR)
if bgr is None:
    raise FileNotFoundError(INPUT_PATH)

h, w = bgr.shape[:2]
# 老师修改：原中心为宽高直接除二；按本题像素坐标中点合同改为各减一再除二。
center = ((w - 1) / 2, (h - 1) / 2)

# 老师修改（以下循环内）：原代码仅测零角度并跨库比较；改为两个角度、同库 A/B 比较。
for angle in [0, 19]:
    # A：先水平翻转，再旋转；每个角度都从原图开始。
    flip_tv = F.hflip(pil_img)
    # 老师修改：原 F.rotate 未指定插值；显式双线性、默认中心、原画布和黑边，固定比较条件。
    rotate_tv = F.rotate(flip_tv, angle, interpolation=InterpolationMode.BILINEAR,
                         expand=False, center=None, fill=(0, 0, 0))
    tv_a = np.array(rotate_tv)
    flip_cv = cv2.flip(bgr, 1)
    # 老师删除：原注释把旋转枚举 0 与角度 0 混在一起；此函数 angle=0 表示不旋转。
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotate_cv = cv2.warpAffine(flip_cv, matrix, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    cv_a = rotate_cv

    # B：先旋转，再水平翻转；保留独立结果，避免覆盖 A 后无法比较。
    rotate_tv = F.rotate(pil_img, angle, interpolation=InterpolationMode.BILINEAR,
                         expand=False, center=None, fill=(0, 0, 0))
    flip_tv = F.hflip(rotate_tv)
    tv_b = np.array(flip_tv)
    rotate_cv = cv2.warpAffine(bgr, matrix, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    flip_cv = cv2.flip(rotate_cv, 1)
    cv_b = flip_cv

    # 老师新增：原代码缺差异量；先分别转有符号整数再相减，避免 uint8 回绕。
    tv_diff = np.abs(tv_a.astype(np.int16) - tv_b.astype(np.int16))
    cv_diff = np.abs(cv_a.astype(np.int16) - cv_b.astype(np.int16))
    print('TorchVision 角度、A/B全图相等、平均绝对分量差：', angle,
          np.array_equal(tv_a, tv_b), round(float(tv_diff.mean()), 6))
    print('OpenCV 角度、A/B全图相等、平均绝对分量差：', angle,
          np.array_equal(cv_a, cv_b), round(float(cv_diff.mean()), 6))

# 老师新增：原提交漏答解释。以下为教师参考，不登记为你独立作答。
# 零角度的旋转不改变图片，因此该特殊情形的一致不能证明任何角度都能交换顺序。
# 本次非零角度的同库 A/B 全图比较否定“谁先谁后总不影响”的说法。
# 固定了原图、角度、中心、插值、画布、补边和库内颜色表示，仅改变操作顺序。
# TorchVision 内部两图都是 RGB，OpenCV 内部两图都是 BGR，无须为同库比较交换颜色。
# 本实验比较顺序能否交换，既没有质量标准，也不能据差异大小给两库排优劣。

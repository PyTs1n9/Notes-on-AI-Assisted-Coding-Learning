import os, cv2  # 导入已有的路径工具和 OpenCV。

import numpy as np  # 导入数组与安全差异计算工具。
from torchvision.transforms import functional as F  # 导入确定执行变换的函数接口。
from torchvision.transforms import InterpolationMode  # 使用明确的双线性插值常量。
from PIL import Image  # 导入 PIL 图片读取接口。

BASE = os.path.dirname(os.path.abspath(__file__))  # 取得脚本目录，跨目录启动也能定位素材。
INPUT_DIR = os.path.join(BASE, 'input')  # 定位本课输入目录。
INPUT_PATH = os.path.join(INPUT_DIR, 'compare_practice.png')  # 定位本课练习原图。
# 老师删除：原 OUTPUT_DIR 从未使用；本题无需保存图片，删掉无效路径配置。

# 老师新增：原提交缺少运行前预测。以下是教师示范假设，不冒充你的原始预测。
# 假设：零角度只剩下翻转，两种顺序应一致；非零角度可能改变结果，需要实际比较。
pil_img = Image.open(INPUT_PATH).convert('RGB')  # 读取为 RGB，作为每次 PIL 组合的共同起点。
bgr = cv2.imdecode(np.fromfile(INPUT_PATH, np.uint8), cv2.IMREAD_COLOR)  # 保留原有中文路径读取方式，解码得到 BGR 数组。
if bgr is None:  # 遇到读取失败先停止。
    raise FileNotFoundError(INPUT_PATH)  # 报告具体输入路径。

h, w = bgr.shape[:2]  # 从输入取得实际高、宽。
# 老师修改：原中心为宽高直接除二；按本题像素坐标中点合同改为各减一再除二。
center = ((w - 1) / 2, (h - 1) / 2)  # 按合同设置像素坐标中点。

# 老师修改（以下循环内）：原代码仅测零角度并跨库比较；改为两个角度、同库 A/B 比较。
for angle in [0, 19]:  # 对合同中的两个角度分别检验，保持每组角度相同。
    # A：先水平翻转，再旋转；每个角度都从原图开始。
    flip_tv = F.hflip(pil_img)  # A 的第一步：对原图水平翻转。
    # 老师修改：原 F.rotate 未指定插值；显式双线性、默认中心、原画布和黑边，固定比较条件。
    rotate_tv = F.rotate(flip_tv, angle, interpolation=InterpolationMode.BILINEAR,  # A 的第二步：旋转翻转结果，明确双线性插值。
                         expand=False, center=None, fill=(0, 0, 0))  # PIL 旋转保持画布、默认中心和黑色补边。
    tv_a = np.array(rotate_tv)  # 单独保留 PIL 路线 A 的 RGB 数组。
    flip_cv = cv2.flip(bgr, 1)  # A 的第一步：对 BGR 原图水平翻转。
    # 老师删除：原注释把旋转枚举 0 与角度 0 混在一起；此函数 angle=0 表示不旋转。
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # 从固定中心、当前角度与原倍率计算旋转矩阵。
    rotate_cv = cv2.warpAffine(flip_cv, matrix, (w, h), flags=cv2.INTER_LINEAR,  # A 的第二步：旋转已翻转的 BGR 图。
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))  # OpenCV 保持常量黑色边界。
    cv_a = rotate_cv  # 保留 OpenCV 路线 A，后续变量重绑定不改此数组。

    # B：先旋转，再水平翻转；保留独立结果，避免覆盖 A 后无法比较。
    rotate_tv = F.rotate(pil_img, angle, interpolation=InterpolationMode.BILINEAR,  # B 的第一步：对 RGB 原图按相同条件旋转。
                         expand=False, center=None, fill=(0, 0, 0))  # PIL 旋转保持画布、默认中心和黑色补边。
    flip_tv = F.hflip(rotate_tv)  # B 的第二步：翻转旋转后的图。
    tv_b = np.array(flip_tv)  # 单独保留 PIL 路线 B 的 RGB 数组。
    rotate_cv = cv2.warpAffine(bgr, matrix, (w, h), flags=cv2.INTER_LINEAR,  # B 的第一步：对 BGR 原图按同一矩阵旋转。
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))  # OpenCV 保持常量黑色边界。
    flip_cv = cv2.flip(rotate_cv, 1)  # B 的第二步：翻转旋转后的 BGR 图。
    cv_b = flip_cv  # 单独保留 OpenCV 路线 B 的 BGR 数组。

    # 老师新增：原代码缺差异量；先分别转有符号整数再相减，避免 uint8 回绕。
    tv_diff = np.abs(tv_a.astype(np.int16) - tv_b.astype(np.int16))  # 两个 RGB 数组各先转 int16，再计算全部分量绝对差。
    cv_diff = np.abs(cv_a.astype(np.int16) - cv_b.astype(np.int16))  # 两个 BGR 数组各先转 int16，再计算全部分量绝对差。
    print('TorchVision 角度、A/B全图相等、平均绝对分量差：', angle,  # 打印当前库和角度，下一行给出数组计算结果。
          np.array_equal(tv_a, tv_b), round(float(tv_diff.mean()), 6))  # 比较本库两种顺序，并计算全图平均差。
    print('OpenCV 角度、A/B全图相等、平均绝对分量差：', angle,  # 打印当前库和角度，下一行给出数组计算结果。
          np.array_equal(cv_a, cv_b), round(float(cv_diff.mean()), 6))  # 比较本库两种顺序，并计算全图平均差。

# 老师新增：原提交漏答解释。以下为教师参考，不登记为你独立作答。
# 零角度的旋转不改变图片，因此该特殊情形的一致不能证明任何角度都能交换顺序。
# 本次非零角度的同库 A/B 全图比较否定“谁先谁后总不影响”的说法。
# 固定了原图、角度、中心、插值、画布、补边和库内颜色表示，仅改变操作顺序。
# TorchVision 内部两图都是 RGB，OpenCV 内部两图都是 BGR，无须为同库比较交换颜色。
# 本实验比较顺序能否交换，既没有质量标准，也不能据差异大小给两库排优劣。

# 第二次考试 · 标准答案（严格按题目要求实现，无额外功能）
# 一、代码填空题：① np.uint8  ② cv2.IMREAD_COLOR  ③ imencode  ④ tofile
#                ⑤ cv2.COLOR_BGR2RGB  ⑥ fromarray  ⑦ cv2.COLOR_RGB2BGR
#                ⑧ p  ⑨ 1  ⑩ path.join
import os
import random
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor


# 1.（20 分）实现带随机角度的旋转增强函数
def random_rotate(img, max_angle=15):
    """getRotationMatrix2D(center, angle, scale)：center 为旋转中心 (x, y)；angle 为旋转角度（度，正值逆时针）；scale 为等比缩放因子"""
    h, w = img.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


# 2.（20 分）实现随机增强组合函数
def random_augment(img):
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.5:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-15, 15), 1.0)
        img = cv2.warpAffine(img, M, (w, h))
    if random.random() < 0.5:
        noise = np.random.normal(0, 10, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


# 3.（30 分）优化离线增强循环保存函数
def process_one(args):
    # 单张图的读取、增强、保存
    f, src_dir, dst_dir, n_aug = args
    try:
        img = cv2.imdecode(np.fromfile(os.path.join(src_dir, f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        name, ext = os.path.splitext(f)
        for i in range(n_aug):
            aug = cv2.flip(img, 1)
            ok, buf = cv2.imencode(ext, aug)
            if ok:
                buf.tofile(os.path.join(dst_dir, f"{name}_flip{i}{ext}"))
    except Exception as e:
        print(f"{f} 处理失败: {e}")


def batch_augment(src_dir, dst_dir, n_aug=5, workers=4):
    # 遍历目录并行调度
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(process_one, [(f, src_dir, dst_dir, n_aug) for f in files]))

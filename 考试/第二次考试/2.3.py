# import os, cv2
#
# src_dir = os.path.dirname(os.path.abspath(__file__))
# src_dir = os.path.join(src_dir, 'src_dir')
# dst_dir = os.path.join(src_dir, 'dst_dir')
#
# for f in os.listdir(src_dir):
# 	img = cv2.imread(os.path.join(src_dir, f))
# 	for i in range(5):
# 		aug = cv2.flip(img, 1)
# 		cv2.imwrite(os.path.join(dst_dir, f'{i}_{f}'), aug)
#

import os
import cv2
from concurrent.futures import ThreadPoolExecutor


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
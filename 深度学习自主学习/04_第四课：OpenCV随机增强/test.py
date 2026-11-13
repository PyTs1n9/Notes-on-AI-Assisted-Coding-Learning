import os, cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE, 'input')
INPUT_PATH = os.path.join(INPUT_DIR, 'practice_aug.png')
OUTPUT_DIR = os.path.join(BASE, 'output_practice')
# 老师修改：原来创建相对目录却按脚本目录保存 → 创建实际保存目录，避免换工作目录后缺目录。
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 老师删除：EXTENSIONS 没有被使用；本题处理指定图片，删除无效变量。

# 老师保留：fromfile 与 imdecode 是原提交已有的中文路径安全读取，本机中文绝对路径 imread 实测失败。
img_decode = cv2.imdecode(np.fromfile(INPUT_PATH, dtype=np.uint8), cv2.IMREAD_COLOR)
if img_decode is None:
	raise FileNotFoundError('File is not found!')
print(img_decode.shape, img_decode.dtype)

# 老师新增：原来只跑一次 → 同进程完整处理两轮，每轮重建同种子生成器，打印来自真实计算。
first_images = []  # 老师新增：在循环前初始化首轮副本容器，确保第二轮引用的变量已定义。
for run_index in range(2):
    rng = np.random.default_rng(37)
    angle = float(rng.uniform(-28, 28))
    alpha = float(rng.uniform(0.85, 1.35))
    # 老师修改：原上界 35 不包含 35 → 改为 36，覆盖题目要求的两端。
    beta = int(rng.integers(-35, 36))
    print('random angle alpha number :', round(angle, 6), round(alpha, 6), beta)

    h, w = img_decode.shape[:2]
    # 老师修改：原中心偏离像素坐标中点 → 宽高先减一再除二，符合本题中心约定。
    center = ((w - 1) / 2, (h - 1) / 2)
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
    # 老师新增：原来漏打裁剪位置与形状 → 输出实际值，使裁剪合同可以验收。
    print('裁剪左、上、宽、高：', x, y, crop_width, crop_height)
    print('裁剪图 shape：', crop.shape)

    resized = cv2.resize(crop, (144, 96), interpolation=cv2.INTER_AREA)
    # 老师修改：原来照搬演示的 5×5、1.1 → 本题核为 7×7，横纵标准差均为 1.4。
    blurred = cv2.GaussianBlur(resized, (7, 7), sigmaX=1.4, sigmaY=1.4, borderType=cv2.BORDER_REFLECT_101)
    # 老师新增：补打最终实际形状与类型，不能只凭参数认定保存尺寸。
    print('最终图 shape、dtype：', blurred.shape, blurred.dtype)

    file_names = ['rotated.png', 'brightness.png', 'crop.png', 'resized.png', 'blurred.png']  # 按处理顺序列出要保存的文件名
    images = [rotated, bright, crop, resized, blurred]  # 同一顺序列出各阶段数组

    if run_index == 0:
        for index in range(len(file_names)):
            save_path = os.path.join(OUTPUT_DIR, file_names[index])
            ok, buf = cv2.imencode('.png', images[index])
            # 老师修改：原来编码失败静默跳过 → 明确报错，成功后仍使用原来的中文路径安全写法。
            if not ok:
                raise RuntimeError('编码失败：' + file_names[index])
            buf.tofile(save_path)
            # 老师新增：原来未读回 → 保留实际通道解码，并逐像素比较整图，不能只检查文件存在。
            saved = cv2.imdecode(np.fromfile(save_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if saved is None:
                raise RuntimeError('读回失败：' + file_names[index])
            same = np.array_equal(saved, images[index])
            if not same:
                raise RuntimeError('像素不一致：' + file_names[index])
            print('读回：', file_names[index], saved.shape, saved.dtype, '像素一致：', same)
        # 老师新增：首轮保存所有阶段的独立副本，用于与第二轮真实重算结果比较。
        first_images = []
        for image in images:
            first_images.append(image.copy())
    else:
        # 老师新增：原来没有同进程重放 → 与第二轮新生成的整图逐阶段比较，而不是把数组与自身比较。
        same_stages = True
        for index in range(len(images)):
            if not np.array_equal(first_images[index], images[index]):
                same_stages = False
        same_final = np.array_equal(first_images[-1], blurred)
        if not same_final or not same_stages:
            raise RuntimeError('同种子重放像素不一致')
        print('同进程重建同种子，最终全图一致：', same_final)
        print('同进程重建同种子，全部阶段一致：', same_stages)

# 老师新增：原提交未回答概念题 → 补充教师参考解释，不计作学生独立作答。
# getRotationMatrix2D 只计算坐标变换矩阵；warpAffine 才把矩阵用于图像，生成新数组。
# 先转 float32 保留负数和超范围值，再 clip 到 0～255，最后转 uint8，避免提前整数转换丢失信息。
# 同一生成器连续抽样会推进状态；重建同种子生成器且保持抽样顺序，才从同一起点重放。

import cv2  # 导入 OpenCV：本例使用 BGR 数组路线
import numpy as np  # 导入 NumPy：把两条路线统一为数组后比较
from PIL import Image  # 导入 Pillow：给 TorchVision 提供 RGB 图片
from torchvision.transforms import functional as F  # 导入直接执行指定操作的函数接口
from torchvision.transforms import InterpolationMode  # 导入插值方式名称

path = 'input/compare_demo.png'  # 使用本课提供的非对称教学图；从本课目录运行
pil_image = Image.open(path).convert('RGB')  # 函数①：明确 PIL 路线为 RGB
bgr = cv2.imread(path)  # 函数②：OpenCV 默认读取为 BGR
if bgr is None:  # 检查输入是否真正读到
    raise FileNotFoundError(path)  # 读取失败时停止，避免拿空对象继续计算

flip_tv = np.array(F.hflip(pil_image))  # 函数③⑥：TorchVision 水平翻转后转为 RGB 数组
flip_cv = cv2.flip(bgr, 1)  # 函数⑦：OpenCV 独立执行水平翻转，结果仍为 BGR
print('未统一颜色的翻转比较：', np.array_equal(flip_tv, flip_cv))  # 函数⑤：故意演示不公平的比较
flip_cv_rgb = cv2.cvtColor(flip_cv, cv2.COLOR_BGR2RGB)  # 函数④：先排除通道顺序这个干扰因素
print('统一颜色后的翻转比较：', np.array_equal(flip_tv, flip_cv_rgb))  # 函数⑤：比较整张图，而非只看形状

resize_tv = F.resize(pil_image, [97, 151], interpolation=InterpolationMode.BILINEAR, antialias=True)  # 函数⑧：尺寸先高后宽
resize_cv = cv2.resize(bgr, (151, 97), interpolation=cv2.INTER_LINEAR)  # 函数⑨：尺寸先宽后高，两库独立计算
resize_tv_rgb = np.array(resize_tv)  # 函数③：将 PIL 结果转为 RGB 数组
resize_cv_rgb = cv2.cvtColor(resize_cv, cv2.COLOR_BGR2RGB)  # 函数④：比较前统一颜色
if resize_tv_rgb.shape != resize_cv_rgb.shape:  # 先排除尺寸顺序错误
    raise ValueError('两条路线的目标尺寸没有对齐')  # 尺寸不同，就不能直接计算逐分量差异
diff = np.abs(resize_tv_rgb.astype(np.int16) - resize_cv_rgb.astype(np.int16))  # 函数⑩：先转换再相减，避免无符号回绕
print('缩放全图相等、平均分量差：', np.array_equal(resize_tv_rgb, resize_cv_rgb), round(float(diff.mean()), 6))  # 函数⑤⑩：用最少证据检验“参数对齐后一定相同吗”

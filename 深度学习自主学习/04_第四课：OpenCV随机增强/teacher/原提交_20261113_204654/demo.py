import os  # 导入已学的路径与目录工具
import cv2  # 导入 OpenCV（图像处理库）
import numpy as np  # 导入 NumPy（数组工具），用 np 简写

input_path = os.path.join('input', 'sample.png')  # 从本课工作目录拼接英文相对输入路径
output_dir = 'output_demo'  # 演示结果使用单独目录
os.makedirs(output_dir, exist_ok=True)  # 目录不存在时创建，重跑时允许目录已存在

image = cv2.imread(input_path, cv2.IMREAD_COLOR)  # 函数⑪：按 BGR 三通道读取图片
if image is None:  # 读取失败时先处理，避免继续访问空结果
    raise FileNotFoundError('无法读取 input/sample.png，请检查本课工作目录')  # 提示工作目录或输入问题
height, width = image.shape[:2]  # 取形状的前两项，先高后宽
print('原图 shape、dtype：', image.shape, image.dtype)  # 打印实际数组形状和元素类型

rng = np.random.default_rng(21)  # 函数①：创建固定种子的随机数生成器
angle = float(rng.uniform(-20, 20))  # 函数②：抽取旋转角度，单位是度
alpha = float(rng.uniform(0.8, 1.3))  # 函数②：抽取线性变换的乘数
beta = int(rng.integers(-25, 26))  # 函数③：抽取加数，整数上界不包含在内
print('随机角度、乘数、加数：', round(angle, 6), round(alpha, 6), beta)  # 只在打印时保留小数位，计算继续使用原值

center = ((width - 1) / 2, (height - 1) / 2)  # 中心按横坐标、纵坐标排列，采用像素坐标中点
matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # 函数④：生成绕中心旋转、大小倍率为 1 的矩阵
print('旋转矩阵：')  # 为下面的矩阵增加说明
print(matrix)  # 打印真实矩阵，它有两行三列

rotated = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))  # 函数⑤：以双线性插值旋转，超出原图处补黑
raw = rotated.astype(np.float32) * alpha + beta  # 函数⑥：先转浮点数再运算，保留负数和超范围的值
print('截断前最小、最大：', float(raw.min()), float(raw.max()))  # min 和 max 汇总所有位置、所有通道的分量

clipped = np.clip(raw, 0, 255)  # 函数⑦：把超出像素范围的分量压回边界
bright = clipped.astype(np.uint8)  # 函数⑥：最后转回图片类型，小数部分直接舍去
print('截断后最小、最大：', int(bright.min()), int(bright.max()))  # 查看最终保存类型中的实际范围

crop_width = 200  # 指定裁剪窗口宽度，不是整张原图的宽度
crop_height = 120  # 指定裁剪窗口高度，必须小于或等于当前图片高度
x = int(rng.integers(0, width - crop_width + 1))  # 函数③：抽取左边界，允许窗口恰好贴住右边
y = int(rng.integers(0, height - crop_height + 1))  # 函数③：抽取上边界，行列位置不要颠倒
crop = bright[y:y + crop_height, x:x + crop_width]  # 函数⑧：先按行再按列切片，结束位置不包含在内
print('裁剪左、上、宽、高：', x, y, crop_width, crop_height)  # 记录随机窗口，便于复现和排查

resized = cv2.resize(crop, (160, 96), interpolation=cv2.INTER_AREA)  # 函数⑨：把裁剪图缩小，尺寸参数先宽后高
blurred = cv2.GaussianBlur(resized, (5, 5), sigmaX=1.1, sigmaY=1.1, borderType=cv2.BORDER_REFLECT_101)  # 函数⑩：用高斯加权平均模糊，边缘使用反射补点
print('裁剪图 shape：', crop.shape)  # 裁剪改变取景范围
print('最终图 shape、dtype：', blurred.shape, blurred.dtype)  # 缩放后再模糊，保持缩放后的形状与类型

file_names = ['rotated.png', 'brightness.png', 'crop.png', 'resized.png', 'blurred.png']  # 按处理顺序列出要保存的文件名
images = [rotated, bright, crop, resized, blurred]  # 同一顺序列出各阶段数组
for index in range(len(file_names)):  # 用已经熟悉的循环逐张保存与读回
    output_path = os.path.join(output_dir, file_names[index])  # 将目录和英文文件名连接起来
    success = cv2.imwrite(output_path, images[index])  # 函数⑫：按 PNG 格式保存 BGR 数组
    if not success:  # 写入失败时不继续假装成功
        raise RuntimeError('保存失败：' + file_names[index])  # 给出失败文件名
    saved = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)  # 函数⑪：按文件实际通道与类型读回
    if saved is None:  # 核对图片能够重新解码
        raise RuntimeError('读回失败：' + file_names[index])  # 给出读回失败文件名
    same = np.array_equal(saved, images[index])  # 函数⑬：逐分量比较整张保存图和内存结果
    if not same:  # 保存成功也必须核对内容
        raise RuntimeError('像素不一致：' + file_names[index])  # 阻止带着内容差异继续运行
    print('读回：', file_names[index], saved.shape, saved.dtype, '像素一致：', same)  # 输出真实的文件形状、类型与比较结果

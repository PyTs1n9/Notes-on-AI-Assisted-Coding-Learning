import os  # 获取脚本位置和输入图片路径。
import sys  # 从调用者取得补卷目录，示范不依赖工作目录。
import cv2  # 解码图像并完成旋转。
import numpy as np  # 读取文件字节并比较图像。

exam_dir = os.path.abspath(sys.argv[1])  # 调用者传入本次检查的补卷目录。
input_path = os.path.join(exam_dir, 'input', 'input.png')  # 路径仍然只是位置字符串。
print('path_is_string', isinstance(input_path, str))  # 验证路径变量的类型。
data = np.fromfile(input_path, dtype=np.uint8)  # 先从文件读取编码后的字节。
img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 再将字节解码为 BGR 图像数组。
if img is None:  # 在读取形状前检查解码是否成功。
    raise ValueError('输入图片解码失败')  # 失败就停止，不能传入旋转函数。
print('decoded_shape', img.shape)  # 输出真正图像数组的高、宽、通道数。
print('decoded_dtype', img.dtype)  # 输出数组的像素存储类型。
original = img.copy()  # 保存原图副本以检查后续操作是否修改输入。
h, w = img.shape[:2]  # 图像数组按高度、宽度的顺序取出尺寸。
center = (w / 2, h / 2)  # 旋转中心按横坐标、纵坐标的顺序填写。
matrix = cv2.getRotationMatrix2D(center, 12, 1.0)  # 正角度旋转，倍率为 1 时不额外缩放。
result = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))  # 画布按宽、高填写。
print('rotated_shape', result.shape)  # 核对固定画布没有改变图像形状。
print('rotated_dtype', result.dtype)  # 核对旋转后仍为 uint8。
print('rotation_changes_pixels', not np.array_equal(result, img))  # 检查配套图片确实发生旋转。
print('input_unchanged', np.array_equal(img, original))  # 验证原图没有被原地改坏。

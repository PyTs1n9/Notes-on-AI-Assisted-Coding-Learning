import os  # 导入已学的 os，用于拼接相对路径和创建目录
import cv2  # 导入 OpenCV 的 Python 接口

input_path = os.path.join('input', 'sample.png')  # 从本课工作目录拼出英文相对输入路径
output_dir = 'output_demo'  # 指定演示专用输出目录，与练习分开
os.makedirs(output_dir, exist_ok=True)  # 创建输出目录，已存在时继续使用
image_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)  # 函数①：读取三通道 BGR 图片
if image_bgr is None:  # 先检查读取结果，避免对 None 查看形状
    raise FileNotFoundError('无法读取 input/sample.png，请检查本课工作目录和输入文件')  # 说明失败原因
print('原图 shape：', image_bgr.shape)  # 打印原图的高、宽、通道
print('原图 dtype：', image_bgr.dtype)  # 打印每个颜色分量的数据类型
print('原图左上角 BGR：', image_bgr[0, 0].tolist())  # 把左上角像素转成列表再打印

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # 函数③：把 BGR 排列转换成 RGB
print('RGB 左上角：', image_rgb[0, 0].tolist())  # 打印同一像素在 RGB 中的分量顺序
restored_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # 函数③：准备交给 OpenCV 保存前，先转回 BGR
resized_bgr = cv2.resize(image_bgr, (200, 100), interpolation=cv2.INTER_AREA)  # 函数④：固定为宽 200、高 100
print('缩放图 shape：', resized_bgr.shape)  # 打印时仍然是高在前、宽在后
print('缩放图 dtype：', resized_bgr.dtype)  # 核对缩放后每个分量的类型

gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)  # 函数③：从缩放后的 BGR 彩图生成灰度图
image_horizontal = cv2.flip(resized_bgr, 1)  # 函数⑤：对缩放图执行左右翻转
image_vertical = cv2.flip(resized_bgr, 0)  # 函数⑤：另外从同一缩放图执行上下翻转
print('灰度图 shape：', gray.shape)  # 灰度图只打印高和宽两个维度
print('灰度图 dtype：', gray.dtype)  # 灰度图依然用 uint8 存亮度值
print('水平翻转 shape：', image_horizontal.shape)  # 翻转不会交换宽高
print('垂直翻转 shape：', image_vertical.shape)  # 上下翻转同样保持形状

copy_path = os.path.join(output_dir, 'sample_copy.png')  # 拼出原图副本的输出路径
rgb_path = os.path.join(output_dir, 'rgb_restored.png')  # 拼出 RGB 转回 BGR 后的输出路径
resize_path = os.path.join(output_dir, 'resized.png')  # 拼出缩放结果的输出路径
gray_path = os.path.join(output_dir, 'gray.png')  # 拼出灰度结果的输出路径
horizontal_path = os.path.join(output_dir, 'horizontal.png')  # 拼出水平翻转结果的输出路径
vertical_path = os.path.join(output_dir, 'vertical.png')  # 拼出垂直翻转结果的输出路径
print('保存原图副本：', cv2.imwrite(copy_path, image_bgr))  # 函数②：写入 BGR 原图并打印真实成功状态
print('保存 RGB 往返图：', cv2.imwrite(rgb_path, restored_bgr))  # 函数②：写入转回 BGR 的彩色数组
print('保存缩放图：', cv2.imwrite(resize_path, resized_bgr))  # 函数②：写入缩放后的图片
print('保存灰度图：', cv2.imwrite(gray_path, gray))  # 函数②：写入二维灰度数组
print('保存水平翻转图：', cv2.imwrite(horizontal_path, image_horizontal))  # 函数②：写入左右镜像结果
print('保存垂直翻转图：', cv2.imwrite(vertical_path, image_vertical))  # 函数②：写入上下镜像结果

saved_gray = cv2.imread(gray_path, cv2.IMREAD_UNCHANGED)  # 函数①：保留文件原有通道方式，读回灰度图
if saved_gray is None:  # 核对文件确实可以重新解码
    raise RuntimeError('灰度文件读回失败')  # 保存状态与可读取状态分别检查
print('读回灰度 shape：', saved_gray.shape)  # 若用彩色读取模式，灰度文件也会被读成三通道
print('读回灰度 dtype：', saved_gray.dtype)  # 打印保存后重新读取的真实类型

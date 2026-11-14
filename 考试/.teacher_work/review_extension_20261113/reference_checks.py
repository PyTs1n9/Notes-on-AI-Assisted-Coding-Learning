import os  # 读取脚本目录，并拼接本次检查需要的路径。
import sys  # 取得调用者传入的补卷目录，不依赖当前工作目录。
import hashlib  # 对比两次运行前后的文件内容是否改变。
import importlib.util  # 从指定补卷目录加载你提交的 process_one.py。
import cv2  # 构建题目规定的水平翻转和旋转结果。
import numpy as np  # 解码磁盘图片并逐像素比较结果。

exam_dir = os.path.abspath(sys.argv[1])  # 命令行第一个参数填写补卷目录。
source_path = os.path.join(exam_dir, 'process_one.py')  # 定位原提交，检查时不修改它。
spec = importlib.util.spec_from_file_location('student_process', source_path)  # 创建加载说明。
student = importlib.util.module_from_spec(spec)  # 建立保存提交函数的模块对象。
spec.loader.exec_module(student)  # 执行定义；模块名不是 __main__，所以不自动调用 main。
input_dir = os.path.join(exam_dir, 'input')  # 输入目录与脚本目录相对定位。
output_dir = os.path.join(exam_dir, 'output_supplement')  # 只检查题目规定的输出目录。
input_path = os.path.join(input_dir, 'input.png')  # 拼出配套原图路径。
input_data = np.fromfile(input_path, dtype=np.uint8)  # 从磁盘读取原图的文件字节。
img = cv2.imdecode(input_data, cv2.IMREAD_UNCHANGED)  # 保留磁盘中的通道数与类型。
if img is None:  # 在使用 shape 前先判断解码是否成功。
    raise ValueError('输入图片解码失败')  # 输入无效时明确停止，不继续变换。
input_hash_before = hashlib.sha256(input_data.tobytes()).hexdigest()  # 保存原图内容指纹。
print('input', 'input/input.png', img.shape, img.dtype)  # 打印本轮实际使用的输入。
first_success, first_failure, first_messages = student.batch_augment(input_dir, output_dir)  # 首次调度。
print('batch_first', first_success, first_failure)  # 统计单位是输入文件，不是产物张数。
for message in first_messages:  # 逐条展示调度返回的处理结果。
    print('message', message)  # 不把失败记录打印成成功。
names = sorted(os.listdir(output_dir))  # 从磁盘枚举实际产物，不直接写死数量。
print('output_names', names)  # 展示真实文件名。
print('output_count', len(names))  # 展示真实文件数。
assert names == ['input_flip.png', 'input_rotate.png']  # 核对文件名与种类。
h, w = img.shape[:2]  # 数组中先取高度，再取宽度。
center = (w / 2, h / 2)  # 坐标中心按横坐标、纵坐标填写。
matrix = cv2.getRotationMatrix2D(center, 12, 1.0)  # 使用规定的角度和不缩放倍率。
expected_rotate = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))  # 生成旋转参照。
expected_flip = cv2.flip(img, 1)  # 生成水平翻转参照。
hashes_before = {}  # 保存首次生成后每个产物的内容指纹。
for name in names:  # 逐张回读，而不是只检查文件是否存在。
    path = os.path.join(output_dir, name)  # 拼出产物的实际路径。
    saved_data = np.fromfile(path, dtype=np.uint8)  # 从磁盘重新取得产物字节。
    saved = cv2.imdecode(saved_data, cv2.IMREAD_UNCHANGED)  # 读回时不强行补成彩色图。
    if saved is None:  # 在访问 shape 前先检查解码结果。
        raise ValueError(name + '：产物解码失败')  # 指明具体失败文件。
    print('artifact', name, saved.shape, saved.dtype)  # 打印磁盘产物的结构。
    assert saved.shape == img.shape and saved.dtype == np.uint8  # 验证实际形状与类型。
    if name == 'input_flip.png':  # 水平翻转有自己对应的像素检查。
        matches = np.array_equal(saved, expected_flip)  # 比较整张图片的全部像素。
    else:  # 另一张产物是固定旋转。
        matches = np.array_equal(saved, expected_rotate)  # 与规定参数生成的结果比较。
    print('pixels_match', name, matches)  # 打印计算得到的核验结论。
    assert matches  # 核验不通过就停止，不能仅打印后继续宣告成功。
    hashes_before[name] = hashlib.sha256(saved_data.tobytes()).hexdigest()  # 保存产物指纹。
second_success, second_failure, second_messages = student.batch_augment(input_dir, output_dir)  # 保留旧图再运行。
print('batch_second', second_success, second_failure)  # 展示本轮拒绝覆盖的统计。
for message in second_messages:  # 展示拒绝覆盖的具体原因。
    print('message', message)  # 不把拒绝覆盖误报成新增产物。
hashes_unchanged = True  # 逐个检查已有产物是否仍为原内容。
for name in names:  # 对第一次生成的每张文件重新取指纹。
    path = os.path.join(output_dir, name)  # 指向原来那张产物。
    data = np.fromfile(path, dtype=np.uint8)  # 重新读取磁盘内容。
    if hashlib.sha256(data.tobytes()).hexdigest() != hashes_before[name]:  # 判断内容有无变化。
        hashes_unchanged = False  # 任一文件改变就记录失败。
input_after = np.fromfile(input_path, dtype=np.uint8)  # 重新读取输入文件。
input_unchanged = hashlib.sha256(input_after.tobytes()).hexdigest() == input_hash_before  # 比较输入内容。
print('output_count_after_second', len(os.listdir(output_dir)))  # 回读实际文件数。
print('existing_hashes_unchanged', hashes_unchanged)  # 输出内容比较结果。
print('input_hash_unchanged', input_unchanged)  # 输出输入保护检查结果。
assert (first_success, first_failure) == (1, 0)  # 此示范要求从无输出目录的副本运行。
assert (second_success, second_failure) == (0, 1)  # 第二次应明确拒绝已有产物。
assert hashes_unchanged and input_unchanged  # 内容与输入保护均须通过。

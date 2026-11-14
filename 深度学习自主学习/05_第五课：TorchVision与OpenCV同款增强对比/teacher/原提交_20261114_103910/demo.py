import os  # 导入路径与目录工具
import cv2  # 导入 OpenCV 图像处理库
import numpy as np  # 导入 NumPy 数组工具
import torch  # 导入 PyTorch，本例只用它管理 TorchVision 的随机源
from PIL import Image  # 导入 Pillow 的图片接口
from torchvision import transforms  # 导入已经接触过的随机变换
from torchvision.transforms import functional as F  # functional（函数接口）直接执行给定参数，不自行抽角度
from torchvision.transforms import InterpolationMode  # 导入 TorchVision 的插值方式名称

input_path = os.path.join('input', 'compare_demo.png')  # 本例使用本课自带的英文相对路径
output_dir = 'output_demo'  # 演示产物与课后练习分开
os.makedirs(output_dir, exist_ok=True)  # 首次运行也能创建输出目录

pil_image = Image.open(input_path).convert('RGB')  # 函数①：打开图片并明确为 RGB
bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)  # 函数②：同一文件在 OpenCV 中按 BGR 读取
if bgr is None:  # 读取失败时不继续处理
    raise FileNotFoundError(input_path)  # 报出本次失败的输入路径

rgb = np.array(pil_image)  # 函数③：把 PIL 图片变为 HWC 的 RGB 数组，没有转成张量
cv_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # 函数④：OpenCV 数组先转换通道顺序再比较
print('输入 PIL size、mode：', pil_image.size, pil_image.mode)  # PIL size 的顺序为宽、高
print('输入数组 shape、dtype：', bgr.shape, bgr.dtype)  # 数组形状先高、后宽、再通道
print('原始数组直接相等：', np.array_equal(rgb, bgr))  # 函数⑤：通道顺序未统一时不能当作内容判断
print('统一 RGB 后相等：', np.array_equal(rgb, cv_rgb))  # 函数⑤：同一无损输入在统一表示后应一致

flip_tv = F.hflip(pil_image)  # 函数⑥：执行确定的水平翻转，没有概率抽样
flip_cv = cv2.flip(bgr, 1)  # 函数⑦：OpenCV 的 1 表示左右翻转
flip_cv_rgb = cv2.cvtColor(flip_cv, cv2.COLOR_BGR2RGB)  # 函数④：转换为 RGB 后再比较
print('水平翻转 RGB 全图相等：', np.array_equal(np.array(flip_tv), flip_cv_rgb))  # 函数③⑤：检查全部位置

resize_tv = F.resize(pil_image, [97, 151], interpolation=InterpolationMode.BILINEAR, antialias=True)  # 函数⑧：尺寸先高后宽；PIL 双线性缩小会抗锯齿
resize_cv = cv2.resize(bgr, (151, 97), interpolation=cv2.INTER_LINEAR)  # 函数⑨：尺寸先宽后高；本例保留库的不同实现
resize_tv_rgb = np.array(resize_tv)  # 函数③：取得 PIL 路线的实际 RGB 数组
resize_cv_rgb = cv2.cvtColor(resize_cv, cv2.COLOR_BGR2RGB)  # 函数④：整理 OpenCV 路线的颜色顺序
resize_diff = np.abs(resize_tv_rgb.astype(np.int16) - resize_cv_rgb.astype(np.int16))  # 函数⑩：先转有符号类型，避免 uint8 相减回绕
print('缩放两图 shape：', resize_tv_rgb.shape, resize_cv_rgb.shape)  # 相同形状不代表逐像素相同
print('缩放全图相等、最大分量差、平均分量差：', np.array_equal(resize_tv_rgb, resize_cv_rgb), int(resize_diff.max()), round(float(resize_diff.mean()), 6))  # 函数⑤：统计全部通道的差异

height, width = bgr.shape[:2]  # 取得输入的高与宽
rng = np.random.default_rng(43)  # 函数⑪：只用这个生成器抽取本次共享角度
angle = float(rng.uniform(-24, 24))  # 函数⑪：先抽一次，再把同一个原值交给两条路线
rotate_tv = F.rotate(pil_image, angle, interpolation=InterpolationMode.BILINEAR, expand=False, center=None, fill=(0, 0, 0))  # 函数⑫：PIL 后端默认围绕图像中心，保持画布
matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1.0)  # 函数⑬：OpenCV 明确使用像素坐标中点
rotate_cv = cv2.warpAffine(bgr, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))  # 函数⑬：执行同角度旋转，不宣称和 PIL 完全等价
rotate_tv_rgb = np.array(rotate_tv)  # 函数③：取得旋转后的 RGB 数组
rotate_cv_rgb = cv2.cvtColor(rotate_cv, cv2.COLOR_BGR2RGB)  # 函数④：先统一通道再计算差异
rotate_diff = np.abs(rotate_tv_rgb.astype(np.int16) - rotate_cv_rgb.astype(np.int16))  # 函数⑩：逐分量计算无回绕的绝对差
print('共享角度：', round(angle, 6))  # 只在显示时舍入，变换使用完整原值
print('旋转两图 shape：', rotate_tv_rgb.shape, rotate_cv_rgb.shape)  # 记录实际画布形状
print('旋转全图相等、最大分量差、平均分量差：', np.array_equal(rotate_tv_rgb, rotate_cv_rgb), int(rotate_diff.max()), round(float(rotate_diff.mean()), 6))  # 函数⑤：差异是观察证据，不是质量排名

tv_angles = []  # 保存两次重建 PyTorch 随机状态后的角度
cv_angles = []  # 保存两次新建 NumPy 生成器后的角度
for repeat_index in range(2):  # 真正重建两次状态，检查各自路线的重复性
    torch.manual_seed(43)  # 函数⑭：重设 PyTorch 随机源，不会替代 NumPy 生成器
    tv_angle = transforms.RandomRotation.get_params([-24, 24])  # 函数⑮：只抽角度，不会变换图片
    local_rng = np.random.default_rng(43)  # 函数⑪：重新创建 NumPy 生成器，两套接口各管各的
    cv_angle = float(local_rng.uniform(-24, 24))  # 函数⑪：相同种子整数不等于跨库相同角度
    tv_angles.append(tv_angle)  # 按先后记录 TorchVision 抽样结果
    cv_angles.append(cv_angle)  # 按先后记录 NumPy 抽样结果
print('各自重置后角度复现：', tv_angles[0] == tv_angles[1], cv_angles[0] == cv_angles[1])  # 实际比较两轮参数
print('相同种子：TorchVision角度、NumPy角度：', round(tv_angles[0], 6), round(cv_angles[0], 6))  # 展示跨随机源参数并不对齐

names = ['flip_tv.png', 'flip_cv.png', 'resize_tv.png', 'resize_cv.png', 'rotate_tv.png', 'rotate_cv.png']  # 命名体现路线和操作
pil_outputs = [flip_tv, resize_tv, rotate_tv]  # 记录 PIL 路线的三张结果
cv_outputs = [flip_cv, resize_cv, rotate_cv]  # 记录 OpenCV 路线的三张结果
for index in range(len(pil_outputs)):  # 每次保存同一操作的两条路线
    tv_path = os.path.join(output_dir, names[index * 2])  # 取得 PIL 路线的目标文件名
    cv_path = os.path.join(output_dir, names[index * 2 + 1])  # 取得 OpenCV 路线的目标文件名
    pil_outputs[index].save(tv_path)  # 函数⑯：PIL 图片本来就是 RGB，直接按 PNG 保存
    success = cv2.imwrite(cv_path, cv_outputs[index])  # 函数⑯：OpenCV 保存的是 BGR，不能错传前面的 RGB 比较数组
    if not success:  # 不把写入失败伪装成成功
        raise RuntimeError('保存失败：' + cv_path)  # 明确失败文件
    for path in [tv_path, cv_path]:  # 两条路线都要从磁盘读回
        saved_bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # 函数②：先检查文件实际通道与类型
        if saved_bgr is None:  # 图片无法解码时立即失败
            raise RuntimeError('读回失败：' + path)  # 明确读回问题
        saved_rgb = cv2.cvtColor(saved_bgr, cv2.COLOR_BGR2RGB)  # 函数④：比较前把读回数组统一为 RGB
        if path == tv_path:  # 选择与保存路线对应的内存结果
            expected_rgb = np.array(pil_outputs[index])  # 函数③：PIL 路线的内存内容
        else:  # OpenCV 路线保存前是 BGR
            expected_rgb = cv2.cvtColor(cv_outputs[index], cv2.COLOR_BGR2RGB)  # 函数④：转换后才能逐像素比较
        same = np.array_equal(saved_rgb, expected_rgb)  # 函数⑤：比较保存前后的整张图片
        if not same:  # 发现存储内容不一致时不继续声称通过
            raise RuntimeError('保存像素不一致：' + path)  # 报出不一致文件
        print('读回：', os.path.basename(path), saved_bgr.shape, saved_bgr.dtype, '像素一致：', same)  # 输出真实读回证据

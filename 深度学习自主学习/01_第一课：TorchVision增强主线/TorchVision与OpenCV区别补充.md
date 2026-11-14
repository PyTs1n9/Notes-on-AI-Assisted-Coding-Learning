# TorchVision 与 OpenCV：区别、转换与选型

## 【本节核心知识点】

TorchVision 更贴近 PyTorch 的模型数据流程，OpenCV 更像通用的图像与视频处理工具箱；最重要的实操差异是数据表示、参数顺序和各自的随机增强方式。

## 【核心概念讲解】

### 1. 先用一句话建立区别

- **TorchVision**：PyTorch 生态里的计算机视觉工具。它的图像变换适合接在模型训练的数据流程里，常把图片变成张量（tensor，张量）后送给神经网络。
- **OpenCV**：通用计算机视觉库，常用于读写图片和视频、几何变换、滤波、颜色处理、摄像头等；不要求后续一定使用 PyTorch。

两者都可以做缩放、翻转、旋转、亮度调整和模糊。不要理解成“一个只能增强，另一个不能增强”。它们的强项和数据接口不同，可以单独使用，也可以组合使用。

### 2. 最大的入门差异：图片在程序里长什么样

| 对比点 | TorchVision 常见输入 | OpenCV 常见输入 |
|---|---|---|
| 常见对象 | PIL 图片或 PyTorch 张量 | NumPy 数组 |
| 颜色通道顺序 | PIL 图片通常是 RGB | `cv2.imread` 默认读成 BGR |
| 张量/数组的通道位置 | 张量通常是 `(C, H, W)` | 数组通常是 `(H, W, C)` |
| 像素类型 | 转张量后常为浮点数，`ToTensor()` 通常缩放到 `[0, 1]` | 读入数组通常为 `uint8`，像素范围 `0` 到 `255` |

`H` 是图片高度，`W` 是宽度，`C` 是颜色通道数。RGB/BGR 都有红、绿、蓝三种通道，区别是排列顺序不同。把 BGR 当成 RGB 显示或输入模型，容易出现红蓝颠倒的偏色。

### 3. 同一操作，两种写法

| 操作 | TorchVision | OpenCV | 容易弄反的地方 |
|---|---|---|---|
| 固定尺寸缩放 | `T.Resize((目标高, 目标宽))` | `cv2.resize(图片, (目标宽, 目标高))` | 参数顺序相反：TorchVision 是高、宽；OpenCV 是宽、高 |
| 水平翻转 | `T.RandomHorizontalFlip(p=1.0)` | `cv2.flip(图片, 1)` | OpenCV 的 `1` 表示左右翻转 |
| 垂直翻转 | `T.RandomVerticalFlip(p=1.0)` | `cv2.flip(图片, 0)` | OpenCV 的 `0` 表示上下翻转 |
| 亮度调整 | `T.ColorJitter(brightness=...)` | `cv2.convertScaleAbs(图片, alpha=..., beta=...)` | `ColorJitter` 会随机抽取变化；OpenCV 这一写法是明确的像素线性变换 |
| 随机旋转/裁剪 | 有现成随机变换，可组合在数据流程中 | 通常自己抽角度/坐标，再调用几何变换 | TorchVision 把随机策略封装好了；OpenCV 给较底层的步骤控制 |

OpenCV 的尺寸参数常写成 `(width, height)`，而 TorchVision 的二元组常写成 `(height, width)`。用正方形尺寸时看不出这个差异；长方形尺寸更容易发现错误。

### 4. 随机性是谁帮你管理

TorchVision 的 `RandomHorizontalFlip`、`RandomRotation`、`RandomResizedCrop` 等变换把随机抽样封装在变换对象内部，适合与数据集读取、批次训练流程配合。

OpenCV 的 `resize`、`flip`、`warpAffine` 等操作通常执行你指定的参数，本身不会替你决定“随机转多少度”或“随机裁哪里”。你可以用 Python 的随机数工具抽参数，再把抽到的参数交给 OpenCV。这样更灵活，但随机策略、边界条件和复现工作需要自己安排。

因此：TorchVision 不是“不能控制”，OpenCV 也不是“没有随机增强”。区别是随机增强的抽样策略通常由谁来组织与管理。

### 5. 怎么选

- 图片增强后直接输入 PyTorch 模型，或需要 `Dataset`/`DataLoader` 里按图片随机变换：优先考虑 TorchVision。
- 需要读视频、摄像头、做传统图像处理、几何操作或通用视觉前后处理：OpenCV 常更顺手。
- 现有代码已经依赖 OpenCV，但模型训练用 PyTorch：可以用 OpenCV 做读取/前处理，再把颜色和维度整理成模型需要的格式；也可以把图片转为 PIL 后使用 TorchVision 变换。
- 不必为了“二选一”重写整个项目。按数据流分工，转换时明确标注颜色顺序、维度顺序和数据类型即可。

## 【核心内置函数拆解】

### ① `cv2.imread(filename, flags)` — 用 OpenCV 读取图片
- **作用**：从文件读取图片并返回 NumPy 数组。
- **完整语法**：`cv2.imread(filename, flags=cv2.IMREAD_COLOR)`。
- **关键参数**：`filename` 是图片路径；默认彩色读取通常得到 BGR；`cv2.IMREAD_GRAYSCALE` 可读成灰度图。
- **返回值**：图片数组；路径不存在或读取失败时会返回 `None`。
- **常见坑**：默认 BGR 而不是 RGB；使用前应检查是否为 `None`。
- **适用场景**：OpenCV 图片处理、视频帧处理与视觉前后处理。

### ② `cv2.cvtColor(src, code)` — 转换颜色空间/通道顺序
- **作用**：按指定转换代码改变图像颜色表示，例如 BGR 转 RGB。
- **完整语法**：`cv2.cvtColor(src, code)`。
- **关键参数**：`src` 是输入数组；BGR 转 RGB 时使用 `cv2.COLOR_BGR2RGB`。
- **返回值**：新的图像数组。
- **常见坑**：不要把 RGB 输入误标成 BGR；仅仅改变变量名不会重排通道。
- **适用场景**：把 OpenCV 读取的图像交给默认按 RGB 解释的 PIL/TorchVision。

### ③ `cv2.resize(src, dsize)` — 用 OpenCV 调整图片尺寸
- **作用**：把图片缩放至指定大小。
- **完整语法**：`cv2.resize(src, dsize, interpolation=cv2.INTER_LINEAR)`。
- **关键参数**：`dsize` 顺序为 `(宽, 高)`；`interpolation` 指定插值方法。
- **返回值**：缩放后的 NumPy 数组。
- **常见坑**：宽高顺序与 `T.Resize((高, 宽))` 相反；插值方法不同会让像素值不完全一样。
- **适用场景**：OpenCV 图片缩放和统一输入大小。

### ④ `cv2.flip(src, flipCode)` — 翻转图片
- **作用**：按指定方向翻转图像。
- **完整语法**：`cv2.flip(src, flipCode)`。
- **关键参数**：`flipCode=1` 为左右翻转，`flipCode=0` 为上下翻转，负数为两个方向都翻转。
- **返回值**：翻转后的数组。
- **常见坑**：`flipCode` 是方向编号，不是概率。
- **适用场景**：明确要求执行某一方向翻转时。

### ⑤ `torchvision.transforms.Resize(size)` 与 `PIL.Image.resize(size)` — TorchVision 与 PIL 缩放
- **作用**：前者用于 TorchVision 变换流程；后者对单张 PIL 图片缩放。
- **完整语法**：`T.Resize(size)`；`image.resize(size)`。
- **关键参数**：TorchVision 的二元组是 `(高, 宽)`；PIL 的二元组是 `(宽, 高)`。
- **返回值**：缩放后的图片对象或张量（取决于输入对象和 TorchVision 版本）。
- **常见坑**：别把两种 API 的尺寸顺序混用；整数尺寸与二元组尺寸含义不同。
- **适用场景**：把 PIL 图片或张量接入 TorchVision 训练预处理流程。

### ⑥ `transforms.ToTensor()` 与 `torch.from_numpy(array)` — 转成张量
- **作用**：把图像数据转换为 PyTorch 可用的张量形式。
- **完整语法**：`T.ToTensor()(pil_image)`；`torch.from_numpy(array)`。
- **关键参数**：`ToTensor` 接收 PIL 图片或受支持的数组；`from_numpy` 接收 NumPy 数组。
- **返回值**：PyTorch 张量。
- **常见坑**：`ToTensor()` 对常见 `uint8` PIL 图像会转为通道在前的浮点张量并缩放至 `[0,1]`；`from_numpy` 通常保留原数组的形状和数据类型，不会自动完成 BGR→RGB、HWC→CHW 或 `/255`。
- **适用场景**：把预处理结果接入 PyTorch 模型；用 OpenCV 时要自己决定并核实通道、维度和数值范围。

## 【完整代码示例 + 逐行注释】

下面让两条路径读取同一张输入图，分别执行相同目标尺寸的缩放和水平翻转。为避免把实现细节混淆成“算法效果不同”，示例比较输出形状、类型和颜色处理流程，不要求缩放后的像素逐点相等：PIL/TorchVision 与 OpenCV 的插值实现可能不同。

```python
import os  # 导入 os，用于从脚本位置定位输入图片
import cv2  # 导入 OpenCV 图像处理库
import torch  # 导入 PyTorch 张量工具
from torchvision import transforms as T  # 导入 TorchVision 变换
from PIL import Image  # 导入 PIL 图片对象

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 定位本脚本所在目录
IMAGE_PATH = os.path.join(BASE_DIR, 'input', 'sample.png')  # 拼出同一张测试图片的路径

cv_bgr = cv2.imread(IMAGE_PATH)  # OpenCV 读取图片，默认通道顺序为BGR
if cv_bgr is None:  # 检查OpenCV是否成功读到文件
    raise FileNotFoundError('OpenCV无法读取图片：' + IMAGE_PATH)  # 读取失败时给出明确错误

cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)  # 将BGR通道顺序转换为RGB
pil_rgb = Image.open(IMAGE_PATH).convert('RGB')  # PIL读取同一图片并统一成RGB

print('OpenCV 原始形状：', cv_bgr.shape)  # 显示OpenCV的高、宽、通道排列
print('OpenCV 数据类型：', cv_bgr.dtype)  # 显示OpenCV像素的数值类型
print('TorchVision 输入类型：', type(pil_rgb).__name__)  # 显示TorchVision变换的PIL输入类型

output_height = 120  # 设置目标高度
output_width = 160  # 设置目标宽度

cv_resized = cv2.resize(cv_rgb, (output_width, output_height))  # OpenCV参数顺序是宽、高
cv_flipped = cv2.flip(cv_resized, 1)  # OpenCV用1表示水平翻转

torch_transform = T.Compose([  # 将TorchVision变换按顺序连接
    T.Resize((output_height, output_width)),  # TorchVision参数顺序是高、宽
    T.RandomHorizontalFlip(p=1.0),  # 把概率设成1，确保本例每次都水平翻转
])  # 结束变换流程定义

torch_pil_output = torch_transform(pil_rgb)  # 对PIL图片执行TorchVision变换
torch_tensor_output = T.ToTensor()(torch_pil_output)  # 转成通道在前、范围通常为0到1的张量

print('OpenCV 输出形状：', cv_flipped.shape)  # 显示OpenCV结果为高、宽、通道
print('TorchVision PIL 输出尺寸：', torch_pil_output.size)  # PIL尺寸按宽、高显示
print('TorchVision 张量形状：', tuple(torch_tensor_output.shape))  # 张量形状按通道、高、宽显示
print('TorchVision 张量类型：', torch_tensor_output.dtype)  # 显示张量数据类型
print('TorchVision 张量范围：', float(torch_tensor_output.min()), float(torch_tensor_output.max()))  # 检查张量像素范围
print('OpenCV 结果颜色通道：RGB')  # 说明OpenCV结果已在上面转换成RGB
```

**真实运行结果**（conda `BaiQing3.10AI` 实测）：

```text
OpenCV 原始形状： (240, 320, 3)
OpenCV 数据类型： uint8
TorchVision 输入类型： Image
OpenCV 输出形状： (120, 160, 3)
TorchVision PIL 输出尺寸： (160, 120)
TorchVision 张量形状： (3, 120, 160)
TorchVision 张量类型： torch.float32
TorchVision 张量范围： 0.1568627506494522 0.9411764740943909
OpenCV 结果颜色通道：RGB
```

### 6. 本例背后的关键转换

如果模型当前需要张量，而图像已经由 OpenCV 读成 BGR 数组，完整转换思路是：先 BGR→RGB，再把通道从 `(H,W,C)` 移到 `(C,H,W)`，再把 `uint8` 转成浮点数并除以 255。对刚入门的流程，先转回 PIL 再用 `T.ToTensor()` 更直观，较不容易漏掉其中一个步骤。

需要注意：数值范围处理取决于后续模型预处理。有的模型还要求减均值、除标准差（Normalize，标准化）；不能因为转成张量就认定模型输入准备完毕。

## 【课后练习】

在同一张图片上，分别用 TorchVision 和 OpenCV 完成非正方形缩放与水平翻转，记录两边的输出尺寸、形状、类型和颜色通道处理过程。

- 用自己的话解释两种 API 的宽高参数顺序为何容易写反。
- 说明 OpenCV 默认读图后，怎样确保送给 TorchVision/PIL 的颜色顺序正确。
- 比较两边输出形状和数据类型；不要求插值像素逐点相等。
- 练习只需提供能运行的代码和真实输出，不需要更改或安装任何环境。

### 我的练习解答

（在此粘贴代码和真实运行结果）

### 我的疑问

（记录没想通的点，会话里问我）

## 【本节重点总结】

1. TorchVision 的随机变换和张量转换适合 PyTorch 训练流程；OpenCV 擅长通用图片、视频和传统视觉处理。
2. 交接时先核对颜色顺序、维度顺序、尺寸参数顺序和数据类型/范围。
3. 选择工具看后续数据流，不是比较谁能做更多种增强；两库可以分工协作。

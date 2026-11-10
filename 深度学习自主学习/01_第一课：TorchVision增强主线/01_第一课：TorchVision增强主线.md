# 第一课：TorchVision增强主线

## 【本节核心知识点】

本课使用 TorchVision 的 `transforms` 组合图像增强流程，重点掌握图片筛选、固定尺寸缩放、翻转、旋转、亮度变化、随机裁剪、模糊、随机种子和输出文件管理。

## 【核心概念讲解】

**图像增强（image augmentation，图像增强）**：对原始图片做缩放、翻转、旋转、亮度变化等处理，生成新的训练图片。它的目的通常是让模型看到更多变化，减少模型只记住原图的风险。

**TorchVision**：PyTorch 生态中专门处理图片和视觉任务的工具库。本课先使用它的 `transforms` 模块完成增强。

一张图片在程序里可以理解为“像素组成的数据”。图片除了宽和高，还有颜色模式。RGB 表示红、绿、蓝三个颜色通道；本课最后统一保存为 RGB，避免颜色通道混乱。

本课的处理链是：

```text
输入目录 → 筛选图片 → 读取并转成 RGB → 依次执行增强 → 保存到输出目录
```

随机增强每次可能产生不同结果。为了让本课代码可以复现，使用固定随机种子。**随机种子（random seed，随机起点**相同，并且随机调用顺序相同，通常就能得到相同的随机结果。

本课还专门放入一个 `说明.txt` 文件，验证程序只处理图片后缀，不会把目录中的所有文件都当成图片读取。

## 【核心内置函数拆解】

### ① `os.path.abspath(path)`、`os.path.dirname(path)` — 获取当前脚本目录

- **作用**：把当前脚本路径转换为绝对路径，再取出所在文件夹。
- **完整语法**：`os.path.abspath(path)`；`os.path.dirname(path)`。
- **关键参数**：`path` 是要处理的路径字符串；`__file__` 表示当前 Python 文件路径。
- **返回值**：返回字符串形式的绝对路径或文件夹路径。
- **常见坑**：相对路径依赖运行时的当前工作目录，换一个运行位置就可能找不到文件；本课用当前脚本目录作为基准。
- **适用场景**：课程脚本需要稳定读取同目录下的输入文件时。

### ② `os.path.join(path, *paths)` — 拼接路径

- **作用**：按照当前操作系统的路径规则拼接多个路径片段。
- **完整语法**：`os.path.join(path, *paths)`。
- **关键参数**：第一个参数是基础目录，后面的参数是子目录或文件名。
- **返回值**：返回拼接后的路径字符串。
- **常见坑**：不要手写 `\\` 或 `/` 来拼接路径；文件名和目录名要分开传入。
- **适用场景**：拼接输入目录、输出目录、图片文件名。

### ③ `os.makedirs(name, exist_ok=False)` — 创建目录

- **作用**：创建目录；如果上级目录不存在，也可以一起创建。
- **完整语法**：`os.makedirs(name, exist_ok=False)`。
- **关键参数**：`name` 是目录路径；`exist_ok=True` 表示目录已存在时不报错。
- **返回值**：成功时返回 `None`。
- **常见坑**：不设置 `exist_ok=True` 时，重复运行脚本可能因为目录已存在而报错。
- **适用场景**：批处理脚本自动准备输入目录和输出目录。

### ④ `os.listdir(path)`、`os.path.isfile(path)` — 遍历并判断文件

- **作用**：`os.listdir` 获取目录中的名称；`os.path.isfile` 判断一个路径是否为文件。
- **完整语法**：`os.listdir(path)`；`os.path.isfile(path)`。
- **关键参数**：`path` 是目录或文件路径。
- **返回值**：`os.listdir` 返回名称列表；`os.path.isfile` 返回布尔值 `True` 或 `False`。
- **常见坑**：`os.listdir` 返回的是名称，不是完整路径；使用前要用 `os.path.join` 拼出完整路径。
- **适用场景**：批量读取文件夹内容，并过滤掉子目录或非目标文件。

### ⑤ `os.path.splitext(path)`、`os.path.basename(path)` — 处理文件名

- **作用**：`splitext` 分离文件主名和后缀；`basename` 取出路径最后的文件名。
- **完整语法**：`os.path.splitext(path)`；`os.path.basename(path)`。
- **关键参数**：`path` 是文件路径字符串。
- **返回值**：`splitext` 返回 `(主名, 后缀)`；`basename` 返回文件名字符串。
- **常见坑**：后缀判断前要统一大小写，例如使用 `.lower()`；否则 `.PNG` 可能被漏掉。
- **适用场景**：筛选图片、生成输出文件名。

### ⑥ `torch.manual_seed(seed)` — 固定 PyTorch 随机种子

- **作用**：设置 PyTorch 的随机数起点，使随机增强更容易复现。
- **完整语法**：`torch.manual_seed(seed)`。
- **关键参数**：`seed` 是整数种子；本课使用 `21`。
- **返回值**：返回 PyTorch 的随机数生成器对象；本课不需要接收它。
- **常见坑**：只固定一个随机库不一定能控制所有随机来源；本课同时固定 `random.seed` 和 `torch.manual_seed`。
- **适用场景**：实验复现、排查随机增强问题、比较不同参数效果。

### ⑦ `transforms.Resize(size)` — 固定尺寸缩放

- **作用**：把图片缩放到指定尺寸。
- **完整语法**：`transforms.Resize(size)`。
- **关键参数**：`size` 可以是整数，也可以是 `(高度, 宽度)`；本课使用 `(160, 160)`。
- **返回值**：返回一个可调用的缩放变换对象。
- **常见坑**：传入二元组时要注意这里使用的是 `(高度, 宽度)`语义；不要和 OpenCV 的 `(宽度, 高度)`参数顺序混淆。
- **适用场景**：模型输入尺寸统一、批量图片预处理。

### ⑧ `transforms.RandomHorizontalFlip(p)` 与 `transforms.RandomVerticalFlip(p)` — 随机翻转

- **作用**：按照概率水平或垂直翻转图片。
- **完整语法**：`transforms.RandomHorizontalFlip(p=0.5)`；`transforms.RandomVerticalFlip(p=0.5)`。
- **关键参数**：`p` 是执行翻转的概率，范围通常是 `0` 到 `1`；本课设为 `1.0`，用于稳定展示翻转效果。
- **返回值**：返回一个可调用的翻转变换对象。
- **常见坑**：`p=1.0` 表示必定执行，`p=0.0` 表示不执行；“随机翻转”不等于每次都翻转。
- **适用场景**：目标左右或上下变化不会改变类别含义的图片。

### ⑨ `transforms.RandomRotation(degrees, fill)` — 随机旋转

- **作用**：在指定角度范围内随机旋转图片。
- **完整语法**：`transforms.RandomRotation(degrees, fill=0)`。
- **关键参数**：`degrees` 是角度范围或最大角度；`fill` 是旋转后空白区域的填充值。
- **返回值**：返回一个可调用的旋转变换对象。
- **常见坑**：旋转后角落可能出现空白区域；填充值的颜色模式要和图片通道匹配。
- **适用场景**：目标方向有一定变化的图片。

### ⑩ `transforms.ColorJitter(brightness)` — 随机亮度变化

- **作用**：随机改变图片亮度，也可以扩展为对比度、饱和度和色调变化。
- **完整语法**：`transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)`。
- **关键参数**：`brightness` 控制亮度变化幅度；本课使用 `0.4`。
- **返回值**：返回一个可调用的颜色变换对象。
- **常见坑**：增强幅度过大可能使图片失真；不同任务允许的颜色变化范围不同。
- **适用场景**：模拟不同光照条件。

### ⑪ `transforms.RandomResizedCrop(size, scale)` — 随机裁剪后缩放

- **作用**：随机截取图片的一部分，再缩放到目标尺寸。
- **完整语法**：`transforms.RandomResizedCrop(size, scale=(0.08, 1.0))`。
- **关键参数**：`size` 是输出尺寸；`scale` 是裁剪区域相对原图面积的范围。
- **返回值**：返回一个可调用的裁剪缩放变换对象。
- **常见坑**：裁剪范围过小可能裁掉目标；输出尺寸和裁剪面积是两个不同参数。
- **适用场景**：模拟目标位置和拍摄范围变化。

### ⑫ `transforms.GaussianBlur(kernel_size, sigma)` — 高斯模糊

- **作用**：使用高斯核让图片变得平滑。
- **完整语法**：`transforms.GaussianBlur(kernel_size, sigma=None)`。
- **关键参数**：`kernel_size` 是模糊核大小；`sigma` 控制模糊强度范围。
- **返回值**：返回一个可调用的模糊变换对象。
- **常见坑**：核大小通常需要符合库的要求；模糊太强会丢失边缘细节。
- **适用场景**：模拟失焦、轻微模糊或提升模型对模糊图片的适应能力。

### ⑬ `transforms.Compose(transforms_list)` — 组合多个变换

- **作用**：把多个变换按列表顺序串起来。
- **完整语法**：`transforms.Compose([变换1, 变换2, ...])`。
- **关键参数**：列表中的顺序就是实际执行顺序。
- **返回值**：返回一个可调用的组合变换对象。
- **常见坑**：顺序会影响结果；例如先裁剪再缩放和先缩放再裁剪不是同一件事。
- **适用场景**：把多个预处理步骤组成完整流水线。

### ⑭ `Image.open(path)`、`convert(mode)`、`save(path)` — 读图、统一模式和保存

- **作用**：`Image.open` 读取图片；`convert("RGB")` 转换颜色模式；`save` 保存图片。
- **完整语法**：`Image.open(path)`；`image.convert(mode)`；`image.save(path)`。
- **关键参数**：`path` 是文件路径；`mode` 是目标颜色模式，本课使用 `RGB`。
- **返回值**：`Image.open` 和 `convert` 返回图片对象；`save` 通常返回 `None`。
- **常见坑**：读取后不统一模式，可能遇到灰度图、RGBA 图与 RGB 图混用；输出扩展名要和保存格式匹配。
- **适用场景**：图片批处理、增强结果落盘。

## 【完整代码示例 + 逐行注释】

```python
import os  # 导入 os，用于处理文件路径、目录和文件遍历
import random  # 导入 random，用于固定 Python 随机数
import torch  # 导入 torch，用于固定 PyTorch 随机数
from PIL import Image, ImageDraw  # 导入图片对象和绘图工具
from torchvision import transforms  # 导入 TorchVision 的图像增强工具

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 函数①：获取本课文件所在目录
INPUT_DIR = os.path.join(BASE_DIR, "input")  # 函数②：拼接输入图片目录
OUTPUT_DIR = os.path.join(BASE_DIR, "output")  # 函数③：拼接输出图片目录
os.makedirs(INPUT_DIR, exist_ok=True)  # 函数④：输入目录不存在时创建目录
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 函数⑤：输出目录不存在时创建目录

random.seed(21)  # 固定 Python 随机数种子，保证本次运行可复现
torch.manual_seed(21)  # 固定 PyTorch 随机数种子，保证随机增强可复现

source_path = os.path.join(INPUT_DIR, "sample.png")  # 函数⑥：拼接示例图片路径
if not os.path.exists(source_path):  # 函数⑦：判断示例图片是否已经存在
    image = Image.new("RGB", (320, 240), (40, 120, 200))  # 函数⑧：创建一张 RGB 示例图片
    draw = ImageDraw.Draw(image)  # 函数⑨：创建绘图对象
    draw.rectangle((40, 50, 160, 190), fill=(240, 180, 60))  # 绘制矩形，帮助观察翻转和裁剪
    draw.ellipse((190, 60, 290, 160), fill=(70, 210, 100))  # 绘制圆形，帮助观察增强结果
    image.save(source_path)  # 函数⑩：把示例图片保存到输入目录

non_image_path = os.path.join(INPUT_DIR, "说明.txt")  # 函数⑪：拼接非图片文件路径
non_image_file = open(non_image_path, "w", encoding="utf-8")  # 函数⑫：以写入模式打开非图片文件
non_image_file.write("这个文件用于测试过滤逻辑。")  # 函数⑬：写入非图片文件内容
non_image_file.close()  # 函数⑭：关闭文件，确保内容落盘

image_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}  # 设置允许处理的图片后缀集合
input_files = []  # 创建空列表，用于保存筛选后的图片路径
for file_name in sorted(os.listdir(INPUT_DIR)):  # 函数⑮：遍历输入目录中的所有名称
    file_path = os.path.join(INPUT_DIR, file_name)  # 函数⑯：拼接当前文件的完整路径
    if os.path.isfile(file_path) and os.path.splitext(file_name)[1].lower() in image_suffixes:  # 函数⑰：判断是否为图片文件
        input_files.append(file_path)  # 把符合条件的图片路径加入列表

print("图片文件数量：", len(input_files))  # 打印过滤后的图片数量

resize = transforms.Resize((160, 160))  # 函数⑱：创建固定尺寸缩放变换
horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)  # 函数⑲：创建必定水平翻转的变换
vertical_flip = transforms.RandomVerticalFlip(p=1.0)  # 函数⑳：创建必定垂直翻转的变换
rotation = transforms.RandomRotation(degrees=20, fill=(0, 0, 0))  # 函数㉑：创建随机旋转变换
brightness = transforms.ColorJitter(brightness=0.4)  # 函数㉒：创建亮度随机变化变换
crop = transforms.RandomResizedCrop(size=(160, 160), scale=(0.8, 1.0))  # 函数㉓：创建随机裁剪后缩放变换
blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))  # 函数㉔：创建高斯模糊变换

transform = transforms.Compose([resize, horizontal_flip, rotation, brightness, crop, blur])  # 函数㉕：按顺序组合多个变换

for file_path in input_files:  # 遍历筛选后的图片文件
    image = Image.open(file_path).convert("RGB")  # 函数㉖：读取图片并统一转换为 RGB
    transformed_image = transform(image)  # 函数㉗：执行组合后的图像增强
    output_name = os.path.splitext(os.path.basename(file_path))[0] + "_augmented.png"  # 函数㉘：生成输出文件名
    output_path = os.path.join(OUTPUT_DIR, output_name)  # 函数㉙：拼接增强结果的输出路径
    transformed_image.save(output_path)  # 函数㉚：保存增强后的图片
    print("已保存：", output_name)  # 打印已保存的文件名

if len(input_files) > 0:  # 判断是否至少处理了一张图片
    print("输出尺寸：", transformed_image.size)  # 打印最后一张增强图片的尺寸
    print("输出模式：", transformed_image.mode)  # 打印最后一张增强图片的颜色模式
else:  # 没有图片时给出提示
    print("没有找到可处理的图片。")  # 打印空目录提示

print("非图片文件已跳过：", os.path.basename(non_image_path))  # 说明非图片文件没有进入处理列表
```

**配套文件**：[code.py](./code.py)

**输入与输出目录**：代码第一次运行时会自动创建 `input/`、`output/`，并在 `input/` 中生成示例图片和非图片文件；增强结果写入 `output/`。

**真实运行结果**（学生环境 `BaiQing3.10AI` 实测）：

```text
图片文件数量： 1
已保存： sample_augmented.png
输出尺寸： (160, 160)
输出模式： RGB
非图片文件已跳过： 说明.txt
```

固定种子连续运行两次，标准输出完全一致；生成文件 `output/sample_augmented.png` 的 SHA-256 两次均为 `799f9dd493751bbd07a608ababbe21315a9ac43a090a8d5f88e3055daab09a1a`。

## 【课后练习】

在本课文件夹中新建 `exercise.py`，完成一个简化版 TorchVision 图片增强脚本。

要求：

1. 使用 `os` 处理脚本目录、输入目录和输出目录。
2. 只处理图片文件，跳过一个你自己放入输入目录的非图片文件。
3. 至少完成固定尺寸缩放、水平翻转、亮度变化和高斯模糊四种增强。
4. 固定随机种子；连续运行两次，并比较两次标准输出和生成图片是否一致。
5. 输出增强图片，打印处理的图片数量、输出文件名、输出尺寸和颜色模式。
6. 代码逐行写注释；将真实运行输出粘贴到本课文档的【我的练习解答】中。

验收点：

- 能成功运行，不因非图片文件报错。
- 输出图片的尺寸符合你设置的目标尺寸，颜色模式为 RGB。
- 连续两次运行的标准输出一致，生成图片的 SHA-256 一致。
- 能在自己的话中解释 `os.path.join`、`os.listdir`、`transforms.Compose` 和随机种子的作用。

> ⚠️ 规矩照旧：作业必须自己在你的 conda **`BaiQing3.10AI`** 环境里实跑，把**真实输出**贴进下面的【我的练习解答】，不贴的退回重交。

### 我的练习解答
```python
import os
import torch
from torchvision import transforms as T
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

os.makedirs(OUTPUT_DIR, exist_ok=True)  # 函数③：自动创建输出目录

torch.manual_seed(42)
image_suffix = {'.jpg', '.jpeg', '.png', '.webp'}
input_files = []
for file_name in sorted(os.listdir(INPUT_DIR)):
	file_path = os.path.join(INPUT_DIR, file_name)
	if os.path.isfile(file_path) and os.path.splitext(file_name)[1].lower() in image_suffix:
		input_files.append(file_path)

transformer = T.Compose([
	T.Resize((160, 160)),
	T.RandomHorizontalFlip(p=1.0),
	T.ColorJitter(brightness=0.5),
	T.GaussianBlur(3, 0.5)
])

print('筛选后图片数量：', len(input_files))  # 显示筛选到的图片数
for file_path in input_files:  # 逐张处理图片
	with Image.open(file_path) as source_image:  # 打开图片并在退出时关闭文件
		image = source_image.convert('RGB')  # 函数⑫：统一成 RGB 图片
	transformed_image = transformer(image)  # 函数⑬：执行组合增强
	output_name = os.path.splitext(os.path.basename(file_path))[0] + '_aug.png'  # 函数⑭：生成新文件名
	output_path = os.path.join(OUTPUT_DIR, output_name)  # 函数②：拼出输出文件路径
	transformed_image.save(output_path)  # 函数⑮：保存增强图片
	print('已保存：', output_name)  # 显示输出文件名
	print('输出尺寸：', transformed_image.size)  # 显示图片尺寸
	print('输出模式：', transformed_image.mode)  # 显示颜色模式

```

```text
筛选后图片数量： 1
已保存： sample_aug.png
输出尺寸： (160, 160)
输出模式： RGB
```

### 我的疑问
’只处理图片文件，跳过一个你自己放入输入目录的非图片文件。‘ 这个怎么做？

## 【本节重点总结】

1. TorchVision 增强是把多个图片变换按顺序组成一条处理流水线。
2. `os` 负责稳定处理目录和文件，图片进入增强前要先筛选并统一为 RGB。
3. 固定随机种子、保存真实输出并检查生成文件，才能证明脚本可复现。


## 老师批改·第1轮（2026-11-10）

**结论：待修正，不结课。** `exercise.py` 与文档所贴代码一致。BaiQing3.10AI 隔离复跑退出码0、stderr为空；两次独立启动 stdout 与图片哈希相同。提交图片也与当前代码复跑生成物哈希相同。但代码没有实现两次结果比较，没有逐行注释或四项概念解释；移除已有 `output/` 后会因输出目录不存在而保存失败。

**本轮对象快照：**

| 文件 | 修改时间 | SHA-256 |
|---|---|---|
| `exercise.py` | 2026-11-10T16:27:23.400000+08:00 | `efd8a6e6921e65d901f75a0c4e27915149feb2c223ec170c8624e26547b91ec9` |
| `01_第一课：TorchVision增强主线.md` | 2026-11-10T16:36:52.849226+08:00 | `e7be537a511b085be6fc9abe2de86f002accf0536504d1de3264dcf2d11ac52f` |
| `input/sample.png` | 2026-11-06T19:21:38.407932+08:00 | `43b791f53acc136ffb6ed0c5d3dcb5c54970b7af1fb41ba070db896b31391b62` |
| `input/说明.txt` | 2026-11-10T16:21:31.070069+08:00 | `92c513d905fd9f8f750aa635f115ba687e8a63a65a5923b9f0f77ddefb6278af` |
| `output/sample_aug.png` | 2026-11-10T16:27:25.387820+08:00 | `eddd195c92c2368c67abd1656f217eae5f3c1409a1b98db0eadd00f6af2ce8b7` |

### 验收结果

| 项目 | 判定 | 证据 |
|---|---|---|
| `os` 路径及图片筛选 | ✅ | 扩展名集合配合 `os.path.isfile`；`说明.txt`未进入图片清单。 |
| 四种变换 | ✅ | Resize、水平翻转、ColorJitter、GaussianBlur均执行成功。`T.Resize(2)`意思是短边缩放到2；本张源图恰好输出2×2，但这不是明确的固定尺寸写法。 |
| 输出文件、尺寸、模式 | ✅ | `sample_aug.png`、`(2, 2)`、RGB；提交图片哈希与复跑相同。 |
| 随机种子 | ✅ | 使用 `torch.manual_seed(42)`。 |
| 程序实际比较两次结果 | ❌ | 只打印“一致”，没有第二次增强、像素比较或哈希比较。教师独立运行两次相同不等于学生代码做了比较。 |
| 逐行注释 | ❌ | exercise.py 没有逐行注释。 |
| 四项概念解释 | ❌ | 没有回答 `os.path.join`、`os.listdir`、`transforms.Compose`、随机种子。 |
| 新目录可运行 | ⚠️ | 未创建输出目录；隔离副本移除 `output/` 后保存报 `FileNotFoundError`。 |
| Markdown 输出围栏 | ✅ 已修复 | 将错误围栏改为标准 `text` 围栏；原输出文字、学生代码、疑问保留。 |

### 原提交教师实跑

第一次真实 stdout（教师捕获，非学生新贴）：

```text
sample_aug.png
(2, 2)
RGB
连续两次生成图片一致
1
```

教师独立运行两次均打印相同内容，图片哈希为 `eddd195c92c2368c67abd1656f217eae5f3c1409a1b98db0eadd00f6af2ce8b7`；仅为教师复验记录，代码没有自行比较。

### 老师答疑：怎样跳过非图片

你的筛选思路正确：`os.listdir` 列目录名称，`os.path.join` 拼出完整路径，`os.path.isfile`排除文件夹，再把后缀转小写并与图片后缀集合比较。只有通过条件的文件才进入 `input_files`，所以 `说明.txt` 不会传给 `Image.open`。本次运行确认它被跳过。

### 错误点参考答案

以下给完整代码和未答概念答案。参考代码只在 scratch 隔离目录运行，没有替换 `exercise.py`；学生源码、作答与疑问原文保留。

#### 可运行参考代码

```python
import os  # 导入 os，处理目录、路径和文件名
import torch  # 导入 torch，固定 TorchVision 随机数
from torchvision import transforms as T  # 导入图像增强工具
from PIL import Image  # 导入图片读取功能

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 函数①：定位脚本目录
INPUT_DIR = os.path.join(BASE_DIR, 'input')  # 函数②：拼出输入目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')  # 函数②：拼出输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 函数③：自动创建输出目录
torch.manual_seed(42)  # 固定当前运行的 Torch 随机种子

image_suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}  # 设置支持的图片后缀
input_files = []  # 创建列表保存图片路径
for file_name in sorted(os.listdir(INPUT_DIR)):  # 函数④：读取并排序目录条目
    file_path = os.path.join(INPUT_DIR, file_name)  # 函数②：拼出完整路径
    suffix = os.path.splitext(file_name)[1].lower()  # 函数⑤：获取小写扩展名
    if os.path.isfile(file_path) and suffix in image_suffixes:  # 函数⑥：只保留图片文件
        input_files.append(file_path)  # 保存符合条件的文件路径
    else:  # 不符合条件的文件走这里
        print('跳过非图片文件：', file_name)  # 显示跳过了什么文件

transform = T.Compose([  # 函数⑦：按顺序组成图像处理流水线
    T.Resize((160, 160)),  # 函数⑧：设置输出高160、宽160
    T.RandomHorizontalFlip(p=1.0),  # 函数⑨：每次都执行水平翻转
    T.ColorJitter(brightness=0.5),  # 函数⑩：随机调整亮度
    T.GaussianBlur(kernel_size=3, sigma=0.5),  # 函数⑪：执行高斯模糊
])  # 结束变换列表

print('筛选后图片数量：', len(input_files))  # 显示筛选到的图片数
for file_path in input_files:  # 逐张处理图片
    with Image.open(file_path) as source_image:  # 打开图片并在退出时关闭文件
        image = source_image.convert('RGB')  # 函数⑫：统一成 RGB 图片
    transformed_image = transform(image)  # 函数⑬：执行组合增强
    output_name = os.path.splitext(os.path.basename(file_path))[0] + '_aug.png'  # 函数⑭：生成新文件名
    output_path = os.path.join(OUTPUT_DIR, output_name)  # 函数②：拼出输出文件路径
    transformed_image.save(output_path)  # 函数⑮：保存增强图片
    print('已保存：', output_name)  # 显示输出文件名
    print('输出尺寸：', transformed_image.size)  # 显示图片尺寸
    print('输出模式：', transformed_image.mode)  # 显示颜色模式
```

真实输出：

```text
跳过非图片文件： 说明.txt
筛选后图片数量： 1
已保存： sample_aug.png
输出尺寸： (160, 160)
输出模式： RGB
```

参考代码两次运行输出一致；生成图均为 `(160, 160)`、RGB，SHA-256 均为 `a07a073602fecd6092f49dfda88bd335ccad5148e042499907e339063785ae76`。

#### 四项概念完整答案

1. **`os.path.join`**：按操作系统规则拼接目录和文件名，得到路径字符串，避免手工判断斜杠方向。这里传入子目录名和文件名；若后一个参数是绝对路径，结果可能改用后者。
2. **`os.listdir`**：列出目录中的名称，包括文件和子目录；不返回完整路径，也不保证排序。常与 `sorted`、`os.path.join`、`os.path.isfile`配合使用。
3. **`transforms.Compose`**：把多个变换按列表顺序连接起来，上一项输出成为下一项输入；不负责读图或保存。
4. **随机种子**：随机数生成器的起点。新进程使用相同种子、相同输入顺序和随机调用顺序通常可复现。同一进程连续调用两次会继续消耗随机状态，结果不一定相同。独立运行复现可启动两次脚本并比图片哈希；脚本内比较同图两次增强，要在两次变换前恢复相同随机状态，再比较图片内容或哈希。

#### 一致性检查的正确做法

不能打印固定的“一致”文字来代替比较。参考代码对每张图片保存随机状态，先增强一次，再恢复该状态增强第二次，并用 `torch.equal` 比较像素张量；通过比较后才保存。每次独立启动脚本也固定种子，可再用输出文件 SHA-256 复核。

### 本轮结论

状态为待修正，不结课、不推送。源码未被修改；当前文档只修正输出围栏并追加批改与完整参考答案。以后提交新版本或要求复验时，检查程序内比较、逐行注释、概念解释及干净目录运行；无需现在立即重交。


## 老师批改·第2轮（2026-11-10，批改对象：最新 exercise.py 与本轮文档作答）

**结论：仍待修正，不推送，不开第二课。** 本轮按最新文件复验；输出目录创建、明确尺寸和部分注释已补上。剩余两项硬要求尚未完成：程序没有真正比较两次增强结果，且四项概念解释仍为空白。代码在BaiQing3.10AI隔离副本连续独立运行两次，退出码均0、stderr为空；stdout一致，保存图片哈希一致。此为教师运行证据。

### 本轮文件快照

| 文件 | 修改时间 | SHA-256 |
|---|---|---|
| `exercise.py` | 2026-11-10T19:24:53.712000+08:00 | `4d52d3f65debf348f6c8b327ef05158e3932a92d07e18480d6a867bb595bf406` |
| `01_第一课：TorchVision增强主线.md` | 2026-11-10T19:25:31.484291+08:00 | `a3f5dd11bcb477c8609599fa77f18e02c18314299401cd84c924656893be9f70` |
| `output/sample_aug.png` | 2026-11-10T19:24:58.461513+08:00 | `a07a073602fecd6092f49dfda88bd335ccad5148e042499907e339063785ae76` |

### 上轮项目逐项复验

| 上轮项目 | 本轮证据 | 结论 |
|---|---|---|
| 创建输出目录 | 当前源码已含 `os.makedirs(OUTPUT_DIR, exist_ok=True)`；隔离运行成功 | ✅ 已修复 |
| 明确固定尺寸 | 当前源码使用 `T.Resize((160, 160))`；实际产物为160×160 | ✅ 已修复 |
| 逐行注释 | 32行可执行代码中，仅12行有行尾注释；其余 import、变量定义、种子、筛选集合、循环及Compose条目等仍未逐行注释 | ❌ 未完成 |
| 程序比较两次结果 | 源码仍无图像比较、哈希比较或断言；主流程仅执行一次增强后保存 | ❌ 未完成 |
| 四项概念解释 | 当前文档没有 `os.path.join`、`os.listdir`、`transforms.Compose`、随机种子的练习文字答案 | ❌ 未完成 |
| 两次独立运行可重复 | 两次 stdout 均为筛选1张、保存 `sample_aug.png`、尺寸`(160,160)`、RGB；SHA-256均为`a07a073602fecd6092f49dfda88bd335ccad5148e042499907e339063785ae76` | ✅ 教师复验通过；不能代替程序内部比较或学生概念回答 |

### 本轮真实实跑输出

以下为最新 `exercise.py` 第一次在隔离副本中的 conda `BaiQing3.10AI` stdout：

```text
筛选后图片数量： 1
已保存： sample_aug.png
输出尺寸： (160, 160)
输出模式： RGB
```

第二次输出逐行相同；两次生成图均为160×160、RGB，哈希相同。提交目录现存 `output/sample_aug.png` 哈希也与本轮运行一致。

### 验收判断

- ✅ 输入图片成功筛选，输出文件真实生成，RGB与明确目标尺寸正确。
- ✅ 输出目录在不存在时自动创建。
- ❌ 一致性要求是程序要验证的行为，不能只靠教师将脚本运行两次后观察相同。当前源码没有实际比较两次结果。
- ❌ 逐行注释不足；建议逐条可执行语句都补注释，尤其导入、路径、种子、扩展名筛选、Compose各项和输出保存。
- ❌ 四项概念解释未作答。第二课开始前仍按原练习要求补齐。

**因此本课仍未通过、未结课；这次源码改动属于向前修正，但不足以触发归档推送。** 原源码不覆盖。当前反馈与完整参考答案见第1轮批改及其后的「错误点参考答案」；本轮没有新增代码修正答案，因为当前版本仍是同一根因——没有真正实现比较。无需立刻重交。

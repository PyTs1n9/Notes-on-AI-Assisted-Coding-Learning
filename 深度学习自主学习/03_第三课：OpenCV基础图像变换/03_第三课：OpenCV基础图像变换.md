# 第三课：OpenCV 基础图像变换

## 【本节核心知识点】

学会用 OpenCV（图像处理库）的 `cv2.imread`、`cv2.imwrite`、`cv2.cvtColor`、`cv2.resize` 和 `cv2.flip`，完成读取、保存、颜色转换、固定尺寸缩放以及水平和垂直翻转；先分清颜色顺序与宽高顺序，再处理图片。

## 【核心概念讲解】

### 1. 前情提要：文件处理继续沿用

上一课已经把找图片、创建输出目录和处理单张图片封装起来，也接触了用真实运行结果核对产物。本课继续用 `os` 拼路径和创建目录，只换一组图像处理工具；本课每次都执行指定的变换，下一课才学习 OpenCV 的随机增强。

### 2. 一张彩色图，在程序里是什么

OpenCV 读取本课的彩色 PNG 图片后，得到 NumPy 数组（按位置存放数字的容器），可以把它理解成一张带颜色信息的数字表。

- **像素**：图片上的一个小点。
- **通道**：一个像素的一种颜色分量；本课的彩色图片有红、绿、蓝三个分量。
- **HWC（高、宽、通道）**：彩色数组的 `shape`（形状）依次记录高度、宽度和通道数。高度是有多少行，宽度是每行有多少个像素。
- **uint8（8 位无符号整数）**：本课图片的 `dtype`（元素类型），每个分量的值在 0～255 之间。不要把“有三个通道”理解成“每个数占三个字节”。

`image[行号, 列号]` 取一个像素，行列编号从 0 开始；`image[0, 0]` 就是左上角。彩色像素会给出三个数字，灰度像素只给出一个数字。演示中的 `.tolist()` 只是把一个像素里的数组转成普通列表，方便打印，不改变图片。

### 3. BGR 与 RGB：相同颜色，排列不同

RGB（红、绿、蓝）和 BGR（蓝、绿、红）说的是**三个颜色分量的排列顺序**。PIL 常用 RGB；本课用 `cv2.imread` 的彩色模式读取，得到 BGR。

例如同一个红色像素，RGB 中记为 `[255, 0, 0]`，BGR 中记为 `[0, 0, 255]`。`cv2.cvtColor` 换了分量的排列，并没有把红色物体染成蓝色。

**数组不会自动记住自己是 RGB 还是 BGR。** 即使 `shape` 和 `dtype` 相同，颜色顺序也可能不同。把 RGB 数组交给按 BGR 解释的 `cv2.imwrite`，通常会出现红蓝互换；保存前要先转回 BGR。普通 PNG 文件没有“这是 OpenCV 的 BGR 文件”这种课程标签，文件里存的是按格式编码的颜色。

### 4. 宽高：看形状与传尺寸是两个顺序

| 位置 | 顺序 | 读法 |
| --- | --- | --- |
| OpenCV 彩色数组的 `shape` | 高、宽、通道 | 先看行数，再看列数 |
| OpenCV 的 `resize` 尺寸参数 | 宽、高 | 先指定横向，再指定纵向 |
| 上一课 PIL 图片的 `size` | 宽、高 | 它也不是数组的形状顺序 |
| 上一课 TorchVision `Resize` 的二元尺寸 | 高、宽 | 不要直接照搬给 OpenCV |

本课演示把输出固定成一个指定的宽高。若目标宽高比例和原图不同，图片会被拉宽或压扁；固定尺寸不等于自动保持原比例。

### 5. 灰度、缩放、翻转各自改变什么

- **灰度图**：每个像素只保留一个亮度值。本课灰度数组的形状是 `(高, 宽)`，不会继续带一个三通道的维度；它不是简单删掉两个颜色分量。
- **缩放**：改变像素的排列范围。插值（估算新位置的像素值）决定新像素如何得到；本课缩小图片用 `INTER_AREA`（区域插值）。
- **水平翻转**：左右镜像，原来的左上角对应翻转后的右上角。
- **垂直翻转**：上下镜像，原来的左上角对应翻转后的左下角。
- 本课用到的转换会返回新数组，原变量不会凭空变成结果；要接住返回值。翻转保持形状与类型，且不是旋转。

## 【核心内置函数拆解】

这里沿用课程的函数卡标题；以下五个函数来自 OpenCV，并非 Python 自带函数。完整语法采用当前 `BaiQing3.10AI` 内 OpenCV 5.0.0 的函数帮助原文：方括号表示可选部分，箭头右侧表示返回值，实际调用时不输入这些说明符号。`src` 表示输入数组，`dst` 表示输出数组；本课不预先提供输出数组，直接接收返回值。

### ① `cv2.imread` —— 从文件读取图片

- **作用**：读取图片并解码成数组。
- **完整语法**：`cv2.imread(filename[, flags]) -> retval`。
- **关键参数**：`filename` 是图片路径；`flags` 是读取方式，本课显式使用 `cv2.IMREAD_COLOR`（三通道彩色读取），需要按文件原样读取时可选 `cv2.IMREAD_UNCHANGED`（保留原有通道等信息）。
- **返回值**：成功时返回图片数组；在本课 Python 环境中读取失败时返回 `None`。
- **常见坑**：读取失败后立刻查看 `shape` 会报错；先检查是否为 `None`。彩色读取结果是 BGR。完整中文路径在本机 OpenCV 实测读取失败，课件用“课程工作目录＋英文相对路径”解决。
- **适用场景**：从本课 `input` 读取演示图或练习图。

### ② `cv2.imwrite` —— 把数组保存成图片文件

- **作用**：按文件扩展名选择编码格式并写入图片。
- **完整语法**：`cv2.imwrite(filename, img[, params]) -> retval`。
- **关键参数**：`filename` 是输出路径，本课使用 `.png`；`img` 是待保存数组，三通道彩色数组按 BGR 解释；`params` 是可选编码参数列表，本课省略。
- **返回值**：写入成功返回 `True`，一般写入失败返回 `False`；不支持的参数或格式等也可能直接抛异常。
- **常见坑**：不会帮你创建父目录；先用已学的 `os.makedirs` 创建目录。会覆盖同名文件，所以演示和练习使用不同目录。同为三通道的 RGB 数组必须先转回 BGR 再保存。
- **适用场景**：保存缩放图、灰度图和翻转图。

### ③ `cv2.cvtColor` —— 转换颜色排列或颜色表示

- **作用**：根据转换代码，把一种颜色表示转换成另一种。
- **完整语法**：`cv2.cvtColor(src, code[, dst[, dstCn[, hint]]]) -> dst`。
- **关键参数**：`src` 是输入图片；`code` 决定转换方向，本课用 `cv2.COLOR_BGR2RGB`、`cv2.COLOR_RGB2BGR` 和 `cv2.COLOR_BGR2GRAY`；`dst` 是可选输出数组，`dstCn` 是可选目标通道数，`hint` 是当前版本的可选算法提示，本课均省略。
- **返回值**：转换后的新数组；BGR 转 RGB 后仍是三通道，BGR 转灰度后在本课得到二维数组。
- **常见坑**：转换代码必须与输入数组的实际颜色顺序一致；输入已经是 RGB，就不能继续当 BGR 来转灰度。较早版本的帮助可能没有 `hint`，本课调用只使用前两个必需参数。
- **适用场景**：与 PIL 等常用 RGB 的工具衔接、把彩色图片变成灰度图片。

### ④ `cv2.resize` —— 缩放到指定尺寸

- **作用**：放大或缩小图片的宽高。
- **完整语法**：`cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) -> dst`。
- **关键参数**：`src` 是输入图片；`dsize` 是目标 `(宽, 高)`；`fx`、`fy` 是横向和纵向缩放比例，在不直接指定目标宽高时可使用；`interpolation` 是插值方法，默认 `cv2.INTER_LINEAR`（双线性插值），本课缩小图片显式选 `cv2.INTER_AREA`。
- **返回值**：缩放后的新数组；本课仍保持三通道和 `uint8`。
- **常见坑**：`dsize` 的宽高顺序与数组形状相反；直接指定目标宽高时，不需要再设置缩放比例；任意宽高组合可能改变原图比例。
- **适用场景**：让后续流程接收到固定尺寸的图片。

### ⑤ `cv2.flip` —— 水平或垂直翻转

- **作用**：按指定方向翻转图片。
- **完整语法**：`cv2.flip(src, flipCode[, dst]) -> dst`。
- **关键参数**：`src` 是输入图片；`flipCode` 为 0 时上下翻转，大于 0 时左右翻转，小于 0 时同时上下和左右翻转。本课用 1 表示水平，用 0 表示垂直。
- **返回值**：翻转后的新数组，形状与元素类型不变。
- **常见坑**：0 是上下，不是左右；`flip` 本身没有随机概率，每次调用都会执行指定翻转；水平与垂直结果都从同一张缩放图分别生成，别误接成连续两次翻转。
- **适用场景**：生成固定方向的镜像图片、检查空间位置变化。

## 【完整代码示例 + 逐行注释】

**运行准备**：继续使用 conda `BaiQing3.10AI`，不需要切换或安装环境。在 PyCharm 的本脚本运行配置中，把“工作目录”设为本课文件夹 `D:/Code/Python/人工智能基础/深度学习自主学习/03_第三课：OpenCV基础图像变换`，再运行 `demo.py`。如果找不到输入，先检查工作目录；本课有意不用含中文的绝对图片路径，避免本机已复现的读取失败。

`input/sample.png` 原样复制自第二课，是随课演示素材，不声称属于真实采集数据集；练习图是另一张教师合成图，来源见 `input/素材说明.md`。本例只使用前者，运行后在 `output_demo` 保存结果；本例重跑会更新本例自己的同名结果。

```python
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
```

**真实运行结果**（教师使用学生同款 conda `BaiQing3.10AI` 实跑，OpenCV 5.0.0、NumPy 2.2.5）：

```text
原图 shape： (240, 320, 3)
原图 dtype： uint8
原图左上角 BGR： [200, 120, 40]
RGB 左上角： [40, 120, 200]
缩放图 shape： (100, 200, 3)
缩放图 dtype： uint8
灰度图 shape： (100, 200)
灰度图 dtype： uint8
水平翻转 shape： (100, 200, 3)
垂直翻转 shape： (100, 200, 3)
保存原图副本： True
保存 RGB 往返图： True
保存缩放图： True
保存灰度图： True
保存水平翻转图： True
保存垂直翻转图： True
读回灰度 shape： (100, 200)
读回灰度 dtype： uint8
```

本次从本文第一个 Python 代码块提取后执行，再自动注入上述输出，`demo.py` 与该代码块完全相同。课程 `.venv` 未安装 `cv2`，因此本课只有上述 conda 环境验证，没有双环境验证，也没有修改任何环境。

阅读结果时留意：RGB 与 BGR 像素的第一个和第三个分量交换；灰度图少了通道维度；翻转不交换宽高。教师另外对所有保存文件做了解码、类型、颜色与空间对应核验，而非只检查文件是否存在。

## 【课后练习】

### 唯一练习：给彩色示意图生成一组固定变换结果

使用本课 `input/practice_rgb.png`，写一个练习脚本。它是教师合成的非正方形彩色示意图：上方左红右蓝，下方左绿右黄，还有不对称标记，便于观察镜像方向；不是现实照片或外部数据集。

**任务步骤**：

1. 用已学的 `os` 管理相对路径；输入沿用本课工作目录约定，输出放在独立的 `output_practice`。目录开始不存在时也能运行。
2. 用 `cv2.imread` 彩色读取练习图并检查读取成功；打印原图的形状、类型和左上角 BGR 像素。
3. 用 `cv2.cvtColor` 得到 RGB 数组，打印左上角 RGB 像素；保存前再转换成 OpenCV 所需的颜色排列，结果文件名为 `color.png`。
4. 用 `cv2.resize` 将原始 BGR 图缩小到验收点要求的固定尺寸，选择区域插值；打印缩放图形状、类型，保存为 `resized.png`。
5. 用 `cv2.cvtColor` 从缩放后的 BGR 图生成灰度图，打印形状、类型，保存为 `gray.png`。
6. 用 `cv2.flip` 分别从同一张缩放图生成水平与垂直翻转图，各打印形状、类型，分别保存为 `horizontal.png`、`vertical.png`；核对每次写入是否成功。
7. 用 `cv2.imread` 保留原有通道的方式读回这些结果，打印每个文件的实际形状和类型，观察红蓝颜色与镜像方向。文字回答：为什么查看数组形状时的宽高顺序，与指定缩放尺寸时不同？为什么 RGB 数组不能直接按 BGR 保存？

**验收点**（目标尺寸和数值只在这里给出）：

- 原图形状为 `(180, 300, 3)`，类型为 `uint8`；左上角 BGR 像素为 `[0, 0, 255]`，转换后的 RGB 像素为 `[255, 0, 0]`。
- 固定输出宽 **180**、高 **108**，插值方法为 `INTER_AREA`；`resized.png` 读回为 `(108, 180, 3)`、`uint8`。
- `color.png` 读回为 `(180, 300, 3)`、`uint8`，全部像素颜色与原图一致，红蓝不互换。
- `gray.png` 保留原有通道方式读回为 `(108, 180)`、`uint8`，内容来自缩放后的彩色图；左上角亮度为 **76**。
- `horizontal.png` 与 `vertical.png` 读回均为 `(108, 180, 3)`、`uint8`；水平图右上角、垂直图左下角都对应缩放图左上角，且整张图分别满足左右镜像、上下镜像关系。
- `output_practice` 内按要求生成上述 **5** 个 PNG，全部可解码，所有保存状态成功；没有预建输出目录也能运行。
- 提交练习代码、完整真实输出与两条概念解释。练习代码不要求逐行注释。

> 在你一直使用的 conda `BaiQing3.10AI` 中运行练习，把代码与完整真实输出放在下面；不需要切换或修改环境。本课不要求随机概率、随机种子、旋转或亮度增强。

### 我的练习解答


### 我的疑问


## 【本节重点总结】

1. OpenCV 彩图默认按 BGR 存放分量，数组形状按高、宽、通道排列，RGB 与 BGR 之间要明确转换。
2. `resize` 接收宽、高，灰度转换改变通道表示，水平与垂直翻转改变像素位置。
3. 保存时检查返回状态并读回核对，确保代码、输出和图片实际内容一致。

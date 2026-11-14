# 第五课：TorchVision 与 OpenCV 同款增强对比

## 【本节核心知识点】

用同一张图片把两条增强路线的颜色、尺寸、参数和随机来源逐项对齐，再用真实全图比较区分“输入没对齐”“随机参数不同”和“实现本身有差异”。

## 【核心概念讲解】

### 1. 前情提要：本课从哪里接上

第四课已教师协助修正后结课：整数抽样上界、旋转中心、模糊参数需要核对，保存后读回及重建同种子验证也已补齐。那次证明修正版产物通过，不等于你已能脱离参考独立组装。本课聚焦公平比较与操作顺序，不再重复上一课的保存读回和重复性检查框架，不额外要求先复述才能开始。单元测试①保留待安排，按你本次决定先开第五课。

### 2. “同款”指同一类任务，不预设所有像素都相同

先建立一张 comparison（对比）表：

| 对比项 | TorchVision本课PIL路线 | OpenCV本课数组路线 | 比较前要做什么 |
|---|---|---|---|
| 输入 | Pillow读取后明确为RGB | 默认彩色读取为BGR | 用同一个无损文件，再统一为RGB |
| 表示 | PIL图片；转NumPy后同样是HWC | NumPy的HWC数组 | 本课不转张量，不提前引入CHW |
| 尺寸 | PIL的size先宽高；F.resize尺寸先高宽 | shape先高宽通道；resize尺寸先宽高 | 用非正方形目标检查顺序 |
| 水平翻转 | F.hflip，确定执行 | flip方向码1，确定执行 | 统一颜色后比较全部像素 |
| 缩放 | PIL双线性缩小及其抗锯齿 | OpenCV INTER_LINEAR | 同名插值不保证相同像素，记录差异 |
| 旋转 | 给定角度，默认图像中心，双线性、补黑、不扩画布 | 同角度，显式像素坐标中点，双线性、补黑、原画布 | 先共用参数；后端坐标、插值和边缘采样仍可能不同 |
| 随机源 | 当前RandomRotation抽角度使用PyTorch | 本课自己用NumPy生成器抽角度 | 相同种子整数不等于两库参数同步 |

本课的“后端”是实际完成计算的实现：TorchVision接收PIL图片时，本例的变换交由Pillow处理；换成张量输入会走另一条实现路线，不能把本课观察无条件推广过去。[S1][S2]

**本课反例提醒**：错：“同一整数种子，两库角度就相同”；错：“两图shape一样，说明内容正确”；错：“像素不同，说明一定有一套库写错”。先检查参数和颜色，再解释差异。

### 3. 为何先比较翻转，再比较缩放与旋转

水平翻转只是重新安排已有像素的位置。它适合当作检查链：读取同一PNG→两路线各自翻转→统一RGB→整图比较。若这一步不同，先查颜色或方向，不急着研究插值。

缩放和任意角度旋转通常需要插值，即根据邻近像素估计新位置的值。抗锯齿指缩小时抑制细小花纹产生的假条纹；PIL双线性缩放会做这类处理，OpenCV本例的INTER_LINEAR没有被我们配置为完全相同的滤波路线。[S1][S3] 本课故意记录差异，不做跨库严格等价复刻，也不凭差异更小判断质量。

PIL默认图像中心与OpenCV像素坐标中点采用不同坐标约定。不能看到宽高除二和减一除二两个写法，就直接宣布两图必定偏移半像素；要结合坐标约定与实测内容判断。本例只把角度、画布、插值类别及补黑方向控制住，不把观测到的旋转差异全部归因给某一个因素。[S2][S3]

### 4. 两种“相同”分开验

- **同一路线可重放**：重新设好对应随机源，保持调用顺序和范围，再跑后得到相同参数与文件。
- **两条路线相同**：同一次比较中，颜色和形状对齐后，比较两库结果。这可能为False，即使两条路线各自完全可重放。

`F`是functional（函数接口）的别名，`F.rotate`只执行你给定的角度；`RandomRotation.get_params`只抽角度，不处理图片。先抽一次共享角度，再交给两库，能排除两次独立抽样带来的差异；本课另设一个实验，展示两套随机源分别重设后各自复现，但相同整数种子仍不对齐。[S4]

### 5. 差异计算先保护类型，保存后再读回

先统一RGB，再把两张uint8数组分别转为int16后相减并取绝对值。int16是可容纳负数的十六位整数，本课只装两幅八位图的分量差；不要在uint8里相减后才转换。

最大分量差表示最不同的某个颜色分量；平均分量差表示整图所有像素、所有通道的差的平均。它们不表示不同像素的个数，也不能用来给两库质量排名。打印时保留小数精度，实际变换参数不提前舍入。

保存时PIL路线保留RGB图片，OpenCV路线保留BGR数组；RGB数组只用于比较。读回每张文件，核对实际shape、dtype和全图像素，避免只验证内存、没验证落地文件。

## 【核心内置函数拆解】

本节主练读取与颜色、翻转、比较和安全差异计算；缩放用于演示，旋转用于迁移练习。随机源和保存路线保留为已学速查，不要求在本次演示或练习重写。这里沿用课程栏目名，主要是库函数和方法，不是Python内置函数。签名按本机BaiQing3.10AI的帮助与官方版本文档核对；方括号表示可选参数，不输入方括号。序号与演示注释对应。

### ① PIL 打开与明确RGB

- **作用**：读取一张图片，并明确后续使用红、绿、蓝的通道顺序。
- **完整语法**：`Image.open(fp, mode='r', formats=None)`；`image.convert(mode=None, matrix=None, dither=None, palette=Palette.WEB, colors=256)`。
- **关键参数**：`fp`为路径或文件对象；`convert`本课只传模式字符串`RGB`，其余参数保留默认。
- **返回值**：`open`返回PIL图片；`convert`返回转换后的图片，不是NumPy数组。
- **常见坑**：图片对象的`size`是宽、高；模式转换不会创建CHW张量，也不自动除以255。只使用已知RGB教学PNG，本课不处理EXIF方向。
- **适用场景**：在TorchVision的PIL路线中读取图片。

### ② cv2.imread

- **作用**：读取输入文件，或读回保存后的真实文件。
- **完整语法**：`cv2.imread(filename[, flags]) -> retval`。
- **关键参数**：`IMREAD_COLOR`读取BGR；`IMREAD_UNCHANGED`保留文件实际通道。
- **返回值**：成功返回图像数组，失败返回`None`。
- **常见坑**：本机中文绝对路径读取已知有问题；本课从课程工作目录使用英文相对路径。读回实际通道后再做颜色转换。
- **适用场景**：OpenCV读取，以及文件落地后的验收。

### ③ np.array

- **作用**：把PIL图片转换成可比较的NumPy数组。
- **完整语法**：`np.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None)`。
- **关键参数**：`object`本课是RGB的PIL图片；`dtype`省略时沿用图片所含类型。
- **返回值**：新的NumPy数组；本课是HWC顺序的RGB、uint8。
- **常见坑**：它既不交换红蓝，也不把HWC变CHW；不要把“转成数组”和“转成模型张量”混在一起。
- **适用场景**：在不引入张量训练的前提下，比较两条图像路线。

### ④ cv2.cvtColor

- **作用**：按指定转换代码改变颜色表示。
- **完整语法**：`cv2.cvtColor(src, code[, dst[, dstCn[, hint]]]) -> dst`。
- **关键参数**：`COLOR_BGR2RGB`用于比较；`COLOR_RGB2BGR`用于将RGB数组交给OpenCV保存。`hint`为当前版本可选算法提示，本课省略。
- **返回值**：转换后的新图像数组。
- **常见坑**：本课不把RGB比较数组送给按BGR约定工作的保存路径；同形状不能发现红蓝错位。
- **适用场景**：比较前统一颜色；跨库传递图片时明确颜色顺序。

### ⑤ np.array_equal

- **作用**：判断形状与全部对应元素是否完全一致。
- **完整语法**：`np.array_equal(a1, a2, equal_nan=False)`。
- **关键参数**：`a1`、`a2`本课必须是同颜色顺序的数组；`equal_nan`沿用默认。
- **返回值**：一个真假值。
- **常见坑**：结果False仅说明有分量不同，不直接说明哪条路线更好；跨库不同不等于同一路线不可重放。
- **适用场景**：检验输入颜色、翻转一致性和保存前后全图内容。

### ⑥ F.hflip

- **作用**：直接执行确定的左右翻转。
- **完整语法**：`F.hflip(img)`，其中`F`是`torchvision.transforms.functional`的别名。
- **关键参数**：`img`本课为PIL图片；没有概率参数。
- **返回值**：本课返回翻转后的PIL图片；也支持文档规定的张量输入。
- **常见坑**：它和随机水平翻转不是同一个调用层级：随机策略需要另外抽样。
- **适用场景**：排除概率干扰，用已知变换检查两条路线是否对齐。

### ⑦ cv2.flip

- **作用**：按翻转方向生成新图。
- **完整语法**：`cv2.flip(src, flipCode[, dst]) -> dst`。
- **关键参数**：`flipCode`取1为左右翻转、0为上下翻转、负数为两方向同时翻转。
- **返回值**：翻转后的数组，形状和类型不变。
- **常见坑**：0不是“不翻转”；先明确是左右还是上下，再比较全图。
- **适用场景**：与确定的水平翻转对照。

### ⑧ F.resize

- **作用**：把图片缩放到指定尺寸。
- **完整语法**：`F.resize(img, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True)`。
- **关键参数**：`size`成对指定时先高后宽；单整数匹配短边并保留比例；本课使用成对尺寸。`interpolation`为双线性；`max_size`本课不用；`antialias`为抗锯齿开关。
- **返回值**：缩放后的图片，本课仍为PIL图片。
- **常见坑**：PIL输入的双线性/双三次缩放始终抗锯齿，不能用`antialias=False`要求它关闭；该开关控制的是相应张量路线。[S1]
- **适用场景**：确定尺寸对比，观察同名插值不保证跨库相同像素。

### ⑨ cv2.resize

- **作用**：将图像数组缩放到指定宽高。
- **完整语法**：`cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) -> dst`。
- **关键参数**：`dsize`先宽后高；本课显式`INTER_LINEAR`；`fx`、`fy`是替代缩放倍率，此处不用。
- **返回值**：缩放后的新数组。
- **常见坑**：不要把PIL size、TorchVision size和NumPy shape的顺序当成同一个；不能只因插值名称相近就要求全图相等。[S3]
- **适用场景**：和PIL双线性路线比较；本课观察差异，不做质量排名。

### ⑩ array.astype、np.abs与差异汇总

- **作用**：先用可容纳负数的类型相减，再统计每个颜色分量的绝对差。
- **完整语法**：`array.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)`；`np.absolute(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature])`，`np.abs`是其别名；`array.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)`；`array.mean(axis=None, dtype=None, out=None, keepdims=False, *, where=True)`。
- **关键参数**：`dtype=np.int16`可容纳本课两幅uint8图像的正负差；汇总时不传`axis`，覆盖高、宽、全部颜色分量。`<no value>`是帮助中的未提供标记，不是实参。
- **返回值**：`astype`返回转换数组，`abs`返回逐分量绝对差，`max`和`mean`返回总体标量。
- **常见坑**：先uint8相减再取绝对值会回绕；本课“平均分量差”是所有像素全部通道差的平均，不是不同像素的数量。
- **适用场景**：报告差异大小；不把差异量误当质量指标。

### ⑪ NumPy生成器与均匀抽样

- **作用**：独立管理随机状态，抽一次共享角度。
- **完整语法**：`np.random.default_rng(seed=None)`；`rng.uniform(low=0.0, high=1.0, size=None)`。
- **关键参数**：`seed`控制初始状态；`low`、`high`为角度上下界；省略`size`每次抽一个。
- **返回值**：前者返回生成器，后者返回浮点数或数组。
- **常见坑**：浮点通常按左闭右开理解，极端舍入可能触及上端；相同种子还要同接口、同版本、同抽样顺序。计算保留原值，打印再舍入。
- **适用场景**：把“抽参数”和“执行变换”拆开，公平观察两条实现。

### ⑫ F.rotate

- **作用**：用给定角度旋转图片，不自行抽取角度。
- **完整语法**：`F.rotate(img, angle, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=None)`。
- **关键参数**：`angle`单位为度，正角逆时针；本课显式双线性；`expand=False`保持画布；`center=None`用后端默认图像中心；`fill`为RGB补充值。[S2]
- **返回值**：旋转后的图片，本课为PIL图片。
- **常见坑**：默认插值是最近邻，需显式改为双线性；后端坐标与边缘采样规则也要考虑，不能只对齐角度就宣称两个实现完全等价。
- **适用场景**：共享抽样角度时执行TorchVision/PIL路线。

### ⑬ OpenCV旋转矩阵与执行

- **作用**：先计算坐标矩阵，再生成输出图片。
- **完整语法**：`cv2.getRotationMatrix2D(center, angle, scale) -> retval`；`cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue[, hint]]]]]) -> dst`。
- **关键参数**：中心先横后纵，本课为宽高各减一后的一半；倍率1；输出宽高保持原图；`INTER_LINEAR`插值、`BORDER_CONSTANT`补BGR黑色。
- **返回值**：前者返回两行三列矩阵，后者返回新图片数组。
- **常见坑**：矩阵不是图片；复制上课演示参数时必须重新核对本题合同。PIL默认中心与OpenCV像素坐标写法不同，本例不把数字坐标不相同直接判成中心偏移。[S2][S3]
- **适用场景**：执行受控的同角度OpenCV旋转。

### ⑭ torch.manual_seed

- **作用**：重设PyTorch随机数状态。
- **完整语法**：`torch.manual_seed(seed)`。
- **关键参数**：`seed`为整数；本课在每次抽样前重设同一个值来重放。
- **返回值**：PyTorch的Generator对象；本课只利用其设置状态的副作用。
- **常见坑**：不会替你重建NumPy default_rng；本课验证当前TorchVision传统transforms抽角度路径，不推广到所有库和所有版本。[S4]
- **适用场景**：对TorchVision抽样来源做可观察的控制。

### ⑮ RandomRotation.get_params

- **作用**：只抽取随机旋转需要的角度。
- **完整语法**：`transforms.RandomRotation.get_params(degrees)`。
- **关键参数**：`degrees`为最小、最大角度组成的序列。
- **返回值**：一个浮点角度，不是图片，也不是旋转矩阵。
- **常见坑**：与`F.rotate`配合才真正旋转；当前已验证版本的该方法使用PyTorch随机源，相同整数种子不保证和NumPy得到同角度。[S4]
- **适用场景**：看清随机增强封装里的“抽参数”阶段。

### ⑯ 两条保存路线

- **作用**：分别把PIL/RGB图片和OpenCV/BGR数组保存为PNG。
- **完整语法**：`image.save(fp, format=None, **params)`；`cv2.imwrite(filename, img[, params]) -> retval`。
- **关键参数**：PIL图片用`.save`；OpenCV数组用`imwrite`；本课由`.png`扩展名决定无损格式；必须提前建父目录。
- **返回值**：PIL的`save`成功返回None，失败抛异常；OpenCV写入成功返回True，也可能抛异常。
- **常见坑**：不能拿PIL的None当作保存失败；也不能只看文件存在，必须读回对照保存前内存结果。
- **适用场景**：同时验证处理结果与文件产物一致。


辅助语法沿用已学的`os.path.join`、`os.makedirs`、`os.path.basename`、列表`append`、循环、`float`、`int`、`round`。InterpolationMode是命名常量的集合，本课只取BILINEAR，无须先学类的实现。

## 【完整代码示例 + 逐行注释】

**本段只回答一个问题：把颜色和尺寸对齐后，两库的变换结果就一定逐像素相同吗？** 水平翻转作为排查颜色问题的基准，缩放用于检查“参数含义相同就必然得到相同像素”的推断。不在这段重复批处理、随机抽样和保存读回框架。

从本课目录运行 `demo_focus.py`，解释器仍为 **BaiQing3.10AI**。原 `demo.py` 含你的本地改动，保持原样，可留作旧流程参考；当前讲义演示以这里和 `demo_focus.py` 为准。

`input/compare_demo.png` 与 `input/compare_practice.png` 是非正方形、不对称的 RGB 教学合成图，包含红蓝区域和细密花纹；练习使用不同素材，来源见 `input/素材说明.md`。

```python
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
```

**真实运行结果**：上面原样代码在现有 BaiQing3.10AI 实跑，输出直接注入：

```text
未统一颜色的翻转比较： False
统一颜色后的翻转比较： True
缩放全图相等、平均分量差： False 5.440295
```

第一行故意保留一个比较条件未对齐的例子，第二行只改颜色表示，第三行再检查缩放。看结果时区分“比较条件不公平”与“条件对齐后实现仍可不同”，无需为每个中间变量打印一行。

**按需查阅**：共享角度、随机源复现和落盘读回已在概念与函数卡中保留；本课不要求重写这些框架。两库旋转存在坐标与插值实现差异，所以新练习先在同一库内部比较顺序，避免把跨库差异误当作顺序影响。

**验证边界**：本轮演示和练习私有参考在现有 BaiQing3.10AI 实跑；课程 `.venv` 原有验证记录为无法导入 cv2，不声称双环境通过，没有改动任何环境。新题不钉死跨版本平均差的小数。

**官方依据**（对应函数卡来源）：

- [S1] https://docs.pytorch.org/vision/0.26/generated/torchvision.transforms.functional.resize.html
- [S2] https://docs.pytorch.org/vision/0.26/generated/torchvision.transforms.functional.rotate.html
- [S3] https://docs.opencv.org/5.0/main_modules/imgproc_transform.html
- [S4] https://docs.pytorch.org/vision/0.26/_modules/torchvision/transforms/transforms.html#RandomRotation.get_params

## 【课后练习】

### 唯一练习：增强菜单没变，只调整顺序，结果就不变吗？

同伴提出：“左右翻转和旋转都做了，谁先谁后不影响结果。”用 `input/compare_practice.png` 设计一个小实验，判断这句话是否可靠，代码写入 `test.py`。本题练的是操作组合与控制比较条件，不是复写演示的跨库流水线。

1. 运行前先写下你的判断和理由，作为待检验的想法；预测错不扣分，运行后根据证据修正。
2. 用已经学过的操作，在 TorchVision 内部比较两种顺序，再在 OpenCV 内部比较两种顺序。每组从相同原图开始，只改变顺序；使用验收点给出的两个角度，观察结论是否需要分情况。
3. 通过整张图的计算比较结果；选择一个已学的差异量说明结果，避免无符号回绕。自己组织循环、变量和输出，只保留能看清“哪个库、哪个角度、顺序是否影响结果”的证据。
4. 用自己的话回答：最初那句话能否成立？为何某个特殊情形不能直接代表一般情形？实验中固定了哪些条件？本实验能否顺便说明哪一个库更好？结合你自己的运行结果解释，不复述题干。

**验收点**：

- 唯一变化是顺序：A 为先水平翻转再旋转，B 为先旋转再水平翻转。分别在 TorchVision 与 OpenCV 内完成 A/B 比较；禁止把另一库的实现差异混入本库顺序比较，也不能拿同一个结果变量冒充两条组合。
- 测试角度为 **0 度和 19 度**；每一组 A/B 使用同一给定角度，不抽随机数。TorchVision 使用 PIL RGB 输入、BILINEAR、默认中心、不扩画布、黑色补边；OpenCV 使用 BGR 输入、像素坐标中点为中心、scale 为 1.0、INTER_LINEAR、原画布、BORDER_CONSTANT 和黑色补边。参数合同固定比较条件，不规定你的代码组织方式。
- 所有结果保持输入画布，实际数组形状为 **(211, 337, 3)**、类型为 **uint8**。比较同库 A/B 的整图相等关系，并选择最大或平均绝对分量差之一作为补充证据；必须由数组真实计算，不打印预设结论。不要求记录每一步 shape 或逐个颜色分量。
- 教师检查实际计算是否与提交结果一致、控制条件是否合理、解释是否回应实验。形状正确不代替像素比较；不以必须匹配某个平均差小数或某种 print 文案为通过条件。
- 本题不要求保存图片、批处理、随机种子实验或两次完整重跑。交代码、一次真实运行结果和简短解释；想看图可自选保存，不新增为验收门槛。练习代码不要求逐行注释。

> 继续使用现有 conda **BaiQing3.10AI**。本题已由老师用私有参考实现验证可完成；题目发布不代表你已掌握，也不提前提供未作答题的答案。

### 我的练习解答

> 2026-11-14教师按明确授权同步为修正版；原源码、原文档和原始四行输出完整保存在[原提交备份](teacher/原提交_20261114_103910/manifest.json)。以下是教师协助修正代码，未登记为本人独立作答。

```python
import os, cv2

import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE, 'input')
INPUT_PATH = os.path.join(INPUT_DIR, 'compare_practice.png')
# 老师删除：原 OUTPUT_DIR 从未使用；本题无需保存图片，删掉无效路径配置。

# 老师新增：原提交缺少运行前预测。以下是教师示范假设，不冒充你的原始预测。
# 假设：零角度只剩下翻转，两种顺序应一致；非零角度可能改变结果，需要实际比较。
pil_img = Image.open(INPUT_PATH).convert('RGB')
bgr = cv2.imdecode(np.fromfile(INPUT_PATH, np.uint8), cv2.IMREAD_COLOR)
if bgr is None:
    raise FileNotFoundError(INPUT_PATH)

h, w = bgr.shape[:2]
# 老师修改：原中心为宽高直接除二；按本题像素坐标中点合同改为各减一再除二。
center = ((w - 1) / 2, (h - 1) / 2)

# 老师修改（以下循环内）：原代码仅测零角度并跨库比较；改为两个角度、同库 A/B 比较。
for angle in [0, 19]:
    # A：先水平翻转，再旋转；每个角度都从原图开始。
    flip_tv = F.hflip(pil_img)
    # 老师修改：原 F.rotate 未指定插值；显式双线性、默认中心、原画布和黑边，固定比较条件。
    rotate_tv = F.rotate(flip_tv, angle, interpolation=InterpolationMode.BILINEAR,
                         expand=False, center=None, fill=(0, 0, 0))
    tv_a = np.array(rotate_tv)
    flip_cv = cv2.flip(bgr, 1)
    # 老师删除：原注释把旋转枚举 0 与角度 0 混在一起；此函数 angle=0 表示不旋转。
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotate_cv = cv2.warpAffine(flip_cv, matrix, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    cv_a = rotate_cv

    # B：先旋转，再水平翻转；保留独立结果，避免覆盖 A 后无法比较。
    rotate_tv = F.rotate(pil_img, angle, interpolation=InterpolationMode.BILINEAR,
                         expand=False, center=None, fill=(0, 0, 0))
    flip_tv = F.hflip(rotate_tv)
    tv_b = np.array(flip_tv)
    rotate_cv = cv2.warpAffine(bgr, matrix, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    flip_cv = cv2.flip(rotate_cv, 1)
    cv_b = flip_cv

    # 老师新增：原代码缺差异量；先分别转有符号整数再相减，避免 uint8 回绕。
    tv_diff = np.abs(tv_a.astype(np.int16) - tv_b.astype(np.int16))
    cv_diff = np.abs(cv_a.astype(np.int16) - cv_b.astype(np.int16))
    print('TorchVision 角度、A/B全图相等、平均绝对分量差：', angle,
          np.array_equal(tv_a, tv_b), round(float(tv_diff.mean()), 6))
    print('OpenCV 角度、A/B全图相等、平均绝对分量差：', angle,
          np.array_equal(cv_a, cv_b), round(float(cv_diff.mean()), 6))

# 老师新增：原提交漏答解释。以下为教师参考，不登记为你独立作答。
# 零角度的旋转不改变图片，因此该特殊情形的一致不能证明任何角度都能交换顺序。
# 本次非零角度的同库 A/B 全图比较否定“谁先谁后总不影响”的说法。
# 固定了原图、角度、中心、插值、画布、补边和库内颜色表示，仅改变操作顺序。
# TorchVision 内部两图都是 RGB，OpenCV 内部两图都是 BGR，无须为同库比较交换颜色。
# 本实验比较顺序能否交换，既没有质量标准，也不能据差异大小给两库排优劣。
```

**老师实跑证据（修正版；不是补造的学生运行记录）**：

```text
TorchVision 角度、A/B全图相等、平均绝对分量差： 0 True 0.0
OpenCV 角度、A/B全图相等、平均绝对分量差： 0 True 0.0
TorchVision 角度、A/B全图相等、平均绝对分量差： 19 False 84.088477
OpenCV 角度、A/B全图相等、平均绝对分量差： 19 False 84.029613
```

#### 老师批改·第1轮（2026-11-14，批改对象：test.py）

原版退出码0、原贴四行输出可复现；但比较对象跨库而非同库A/B，漏19度、差异量、明确双线性插值/像素中心合同与文字解释。本轮已按授权保存原件、直接修正并完成全图核验，**教师协助修正后通过，本课作业关闭**；独立掌握状态不提升。[老师批改、全部错点参考答案与真实输出](teacher/第五课_教师批改_2026-11-14.md)。

### 我的疑问

（记录没想通的点，会话里问我）

#### 老师答疑（2026-11-14）

本轮原问题、具体改法、完整参考解释与真实输出见[错误点参考答案](teacher/第五课_教师批改_2026-11-14.md#3-错误点参考答案)。关键是比较同库两种顺序：0度两者一致，19度两者不同；特殊情形不能推出所有角度都相同。枚举值0和角度0也需区分；这次不以两库差异大小判库优劣。以上预测和解释为教师参考，不补记本人独立掌握。

## 【本节重点总结】

先把同一输入的颜色、维度和参数合同对齐，再判断像素差异。抽参数和执行变换是两步，同一路线可重放与两库结果相同也要分别检查。差异值只能描述当前两条路线的不同，不能直接给库的质量排名。

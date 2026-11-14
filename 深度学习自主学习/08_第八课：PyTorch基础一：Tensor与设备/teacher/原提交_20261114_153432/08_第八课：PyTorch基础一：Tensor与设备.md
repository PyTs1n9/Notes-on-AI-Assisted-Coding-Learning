# 第八课：PyTorch基础一：Tensor与设备

> 2026-11-14开课。第五课已教师协助修正后通过并完成双远程归档；按既定安排05后直接接08，06、07编号留空且不记通过。当前课件与演示已验证，课后练习待提交，不登记独立掌握。
> 学生继续使用 `D:/Conda_envs/envs/BaiQing3.10AI/python.exe`。本机学生环境 PyTorch 为 `2.11.0+cu128`；教师 `.venv` 缺少 `torch`，所以本文件不能标成“双环境验证通过”。下方真实输出来自学生解释器运行本文件确切代码；未安装或改动任何环境。

## 【本节核心知识点】

看懂 Tensor（张量）的 shape（形状）、dtype（元素类型）与 device（设备），能保住像素含义完成 HWC→CHW→BCHW，并区分重新分组、换轴和数值运算。

## 【核心概念讲解】

### 1. 从数组走到张量

**前情提要**：第五课修正版在同一库内比较两种顺序：零角度一致、非零角度不同。原提交把比较对象放到了两库之间；这次迁移时先说清“比较谁、要保持什么不变”。图像增强完成的是数据准备，不等于已掌握模型训练；本课从张量数据表示开始。

PyTorch 是进行张量计算、搭建神经网络的软件库。Tensor（张量）可以先理解为“按多个方向排好的数字”：一个数没有轴，一排数有一个轴，一张数表有行轴和列轴；图像再加上通道轴。它可以保存输入、输出和模型参数，本课只学习数据表示与运算，不训练模型。[10]

axis（轴）是“沿哪个方向取数”，dimension（维度）在本课指轴；维度编号从零开始。`shape` 记录每根轴的长度，不记录“这个轴叫高度还是通道”，语义需要我们自己维护。`x.shape`、`x.dtype`、`x.device` 是属性，读取时没有括号。

`dtype` 决定每个元素怎样存储：`torch.uint8` 是不带负号的八位整数，常用于原始像素；`torch.float32` 是三十二位浮点数，能表达小数和负数；`torch.int64` 是六十四位有符号整数，后续常用于类别编号。[8] 本课数据创建时明确写类型和CPU设备，避免依赖默认设置。

**构造图像张量不等于归一化。** `torch.tensor` 复制传入数字；`.to` 改类型或设备，不会自动按图像习惯缩放取值。[1][5] 归一化是按明确规则调整数值范围；例如除以像素上限是一种缩放，减均值再除标准差是另一种规则，本课不把两者混为一谈，也不提前引入完整预处理管道。

### 补充答疑：这组 `data` 怎样读？

把嵌套列表从外到内读，正好对应 H、W、C 三根轴：

- 最外层有 2 个元素：第 0 行和第 1 行，所以高度 H 为 2。
- 每一行有 2 个内层列表：每一行有 2 个像素，所以宽度 W 为 2。
- 每个最内层列表有 3 个数字：一个像素有 R、G、B 三个颜色分量，所以通道数 C 为 3。

例如，`data[0]` 是第 0 行，内容为 `[[11, 21, 31], [12, 22, 32]]`；其中 `data[0][0]` 是第 0 行第 0 列的像素，RGB 为 `[11, 21, 31]`，`data[0][1]` 是同一行第 1 列的像素，RGB 为 `[12, 22, 32]`。因此这里确实是“第 0 行有 2 个像素，第 1 行也有 2 个像素”，完整数据形状是 2 行 × 2 列 × 3 个颜色分量，即 `(2, 2, 3)`。

`data` 本身只是 Python 的嵌套列表，主要负责把数字装起来；它没有 PyTorch Tensor 的 `dtype`、`device`、`shape` 属性，也不能直接使用本课的 `permute`、`reshape`、`.to` 等张量操作。`torch.tensor(data, dtype=torch.uint8, device='cpu')` 会根据这份嵌套关系创建一个 Tensor：实跑得到 `torch.Size([2, 2, 3])`、`torch.uint8`、`cpu`。这一步没有改变 HWC 顺序、没有把 RGB 改成 BGR，也没有把像素缩放到 0～1；它只是把普通数字容器变成 PyTorch 能够进行张量计算、换轴、设备搬运并作为模型输入的对象。

### 2. 图像的轴名与索引

H（高度）对应行，W（宽度）对应列，C（通道）对应颜色分量；RGB（红绿蓝）是通道内部的颜色顺序。B（批次）表示同一次送入的一组样本，BCHW也常写作NCHW，N同样表示样本数量。HWC、CHW、BCHW写的是轴顺序；RGB与BGR写的是颜色顺序，两个问题要分开。换轴不会自动做BGR→RGB。

| 表示 | 每根轴的含义 | 读取同一像素的颜色分量 |
|---|---|---|
| HWC | 行、列、通道 | `x[h, w, :]` |
| CHW | 通道、行、列 | `x[:, h, w]` |
| BCHW | 样本、通道、行、列 | `x[b, :, h, w]` |

`:` 表示该轴全取；整数索引选定一个位置，会去掉对应轴；切片保留该轴。起点包含、终点不包含。本课只用整数与切片，不引入高级索引。

### 3. 重新分组、换轴、增加轴

`reshape`（重塑形状）按张量当前逻辑遍历顺序重新分组，保持元素总数；它不认识轴名，不会因为目标首轴长度像通道数，就自动把原来分散的同色通道收集起来。`permute`（重排轴）明确指定“新轴依次取自哪些旧轴”，所以需要保持像素坐标含义的HWC→CHW要按轴关系处理。[2][3]

`unsqueeze`（插入单元素轴）只插入一根长度为一的轴；它能给一张CHW图补批次轴，但没有复制出多张样本。[4] `permute`、`unsqueeze`会共享底层数据，`reshape`可能共享也可能复制；不要靠它是否复制来判断结果对错，本课不修改这些共享结果。[2][3][4]

### 4. 基本运算与“能算但算错方向”

`+`、`-`、`*`、`/` 分别做逐元素加、减、乘、真除；`*`不是矩阵乘法。标量（单个数字）可以作用于所有元素；同形张量则按位置运算。保存运算结果时使用新变量，便于比较前后数据。

broadcasting（广播）是让尺寸兼容的张量自动对齐后运算：从最右侧轴向左对齐，对应长度相同或其中一方为一才兼容；短的一侧可以补轴。[9] 例如，一个一维偏移向量会对齐到输入的最后一根轴，代码能运行并不保证它对齐的是“通道”。不要只凭输出shape判断。

整数运算也受类型范围限制；不带负号的整数无法直接存负值。对可能产生负数的计算，要先判断计算时所需类型，计算结束后再改类型不保证能补救已经发生的整数范围问题。不同dtype混合计算可能自动提升类型，本课先让两个操作数具有相同的显式类型，排除这个干扰。练习会让你亲自调查转换与计算的先后顺序。

### 5. CPU/GPU与返回值

CPU（中央处理器）和GPU（图形处理器）是执行计算的设备；CUDA是PyTorch使用兼容NVIDIA GPU的计算接口。`torch.cuda.is_available`检测当前运行环境能否使用CUDA；`torch.device`只描述目标设备，创建这个描述本身不代表搬动数据或保证设备可用。[7][8]

`.to`把需要的类型或设备转换结果返回给你，接住返回值后继续计算。原来已经是目标类型和设备时，可能直接返回原张量；不要假设它每次都复制。[5] 本课统一先在CPU完成概念核验，设备搬运单独看；向量、图像等普通运算输入应放到同一设备，不把特殊标量例外当作通用写法。[8]

## 【核心内置函数拆解】

以下签名由本机已安装PyTorch的文档字符串、可取得的签名及官方2.11文档交叉核对；本节沿用“内置函数”栏目名，实际主要是库函数、张量方法和访问语法。语法里的 `x` 是已有张量，`input` 是输入参数名；`*` 表示后续参数须按名字填写，`None` 表示不另指定，`False` 表示关闭选项。

### ① `torch.tensor`：从已有数字创建张量

- **作用**：复制列表、数组或标量中的数字，构造张量。[1]
- **完整语法**：`torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False)`。
- **关键参数**：`data`是源数据，嵌套列表各层需整齐；`dtype`指定元素类型；`device`指定设备。`requires_grad`控制是否记录用于求导的运算，`pin_memory`是CPU内存的一项传输优化，本课均保持关闭，不展开求导或优化。
- **返回值**：一个Tensor。
- **常见坑**：不自动换轴、改RGB/BGR或缩放像素；dtype省略会根据输入推断。
- **适用场景**：用少量确定数字观察张量规律，或把已有数组作为张量输入。

### ② 索引、切片与属性：先明确取的是哪个轴

- **作用**：读取元素、局部张量以及形状、类型、设备信息。
- **完整语法**：`x[i]`、`x[start:stop:step]`、`x[i, j, :]`；属性为 `x.shape`、`x.dtype`、`x.device`。这是访问语法，不是假装存在一个索引工具函数。
- **关键参数**：位置从零计数；`start`为起点、`stop`为不包含的终点，`step`为步长，本课使用正步长；`:`保留全轴。
- **返回值**：索引返回Tensor，单元素也仍是零维Tensor；属性分别返回形状描述、类型对象、设备对象。
- **常见坑**：整数索引会少一根轴；同一个索引序列放在HWC和CHW上含义不同。
- **适用场景**：追踪同一像素，检查换轴前后是否仍是原来的颜色分量。

### ③ 运算符：逐元素计算

- **作用**：对对应位置计算加减乘除，或把标量应用到张量元素。
- **完整语法**：`x + y`、`x - y`、`x * y`、`x / y`。
- **关键参数**：`x`、`y`是张量或标量；两个张量要满足广播规则，普通非标量输入设备需一致。[9][8]
- **返回值**：计算结果Tensor；本课显式用float32计算，不靠隐式类型提升猜dtype。
- **常见坑**：`*`不是矩阵乘；广播成功可能只是长度碰巧相等；整数结果受可表示范围限制。
- **适用场景**：调整数值、做差，或对通道施加不同偏移。

### ④ `torch.reshape`：保持元素数量，改变分组

- **作用**：按指定形状重新分组相同数据。[2]
- **完整语法**：`torch.reshape(input, shape)`。
- **关键参数**：`input`为张量；`shape`为目标形状元组，元素总数须不变，最多一个位置可写 `-1` 让PyTorch推断。
- **返回值**：目标形状Tensor；可能共享原数据，也可能复制。
- **常见坑**：形状合法不等于轴语义正确；不能用改形状的方式假装完成换轴。
- **适用场景**：按已确认的顺序展平或重新分组；不是根据目标尺寸猜颜色位置。

### ⑤ `torch.permute`：按照旧轴编号重排轴

- **作用**：重新排列各根轴，保留各位置对应的数据含义。[3]
- **完整语法**：`torch.permute(input, dims)`。
- **关键参数**：`dims`依次列出新轴来自哪根旧轴；必须覆盖全部轴，每根旧轴恰好出现一次。本课只用非负编号。
- **返回值**：与原张量共享数据、轴顺序改变的Tensor。
- **常见坑**：`dims`是旧轴编号，不是目标长度；改变轴顺序不等于改变RGB/BGR顺序。
- **适用场景**：HWC与CHW之间转换，或把明确的轴语义换到新位置。

### ⑥ `torch.unsqueeze`：插入长度为一的轴

- **作用**：在指定位置新增一根单元素轴。[4]
- **完整语法**：`torch.unsqueeze(input, dim)`。
- **关键参数**：`dim`为插入位置；已有n根轴时，非负位置可取零至n，负位置可取 `-n-1` 至 `-1`。
- **返回值**：共享原数据的新Tensor，轴数增加一。
- **常见坑**：补批次轴不等于把输入复制成很多样本；插在错误位置会改变轴含义。
- **适用场景**：把单张图像补成带批次轴的表示，或为广播显式对齐轴。

### ⑦ `torch.equal`：比较整个张量的形状与值

- **作用**：整体判断两个张量是否同形且所有对应元素相等。[6]
- **完整语法**：`torch.equal(input, other)`。
- **关键参数**：`input`和`other`为待比较Tensor；本课先放同一设备再比较。
- **返回值**：Python布尔值 `True` 或 `False`。
- **常见坑**：它不负责保证dtype相同，要另查 `.dtype`；有NaN（缺失/无效数值标记）的张量不会按这种方式相等；后续复杂小数运算可能要用近似比较，本课使用能精确表示的小数据。
- **适用场景**：用完整数据比较驳斥“shape一样就一样”的判断。

### ⑧ `torch.cuda.is_available`：检查当前CUDA可用性

- **作用**：询问当前运行环境是否能使用CUDA。[7]
- **完整语法**：`torch.cuda.is_available()`。
- **关键参数**：无。
- **返回值**：Python布尔值。
- **常见坑**：安装了带CUDA名字的包，不等于检测必为真；检测为假时不得强制搬到CUDA。
- **适用场景**：选择可运行设备；没有CUDA时继续在CPU完成本课。

### ⑨ `torch.device`：描述目标设备

- **作用**：创建一个描述设备类型和可选编号的对象。[8]
- **完整语法**：本课使用 `torch.device(device_string)`；官方还支持 `torch.device(device_type, device_index)` 和 `torch.device(device_index)`，最后一种依赖当前加速器，本课不使用。
- **关键参数**：`device_string`可写 `'cpu'`、`'cuda'`、`'cuda:0'`；`device_type`为设备类型，`device_index`为设备编号；创建CUDA对象不验证设备能实际分配张量。
- **返回值**：`torch.device`对象；其 `.type` 给出设备类型。
- **常见坑**：只生成device变量不会迁移任何已有张量；不应仅因电脑有显卡就强制选CUDA。
- **适用场景**：在检查可用性后，把明确的设备目标传给张量转换。

### ⑩ `Tensor.to`：转换类型或设备并接住返回值

- **作用**：返回具有指定dtype和/或device的张量。[5]
- **完整语法**：`x.to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format)`；`x.to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format)`；`x.to(other, non_blocking=False, copy=False)`。本课实际只使用 `x.to(dtype=...)` 与 `x.to(device=...)` 两种明确命名的形式。
- **关键参数**：`dtype`指定类型；`device`指定设备；`other`表示跟另一个张量的类型和设备一致。`non_blocking`涉及异步传输，`copy`要求即使类型设备相同也复制，`memory_format`控制内存格式；本课保持默认，不把它们加入练习。
- **返回值**：转换结果Tensor；已经符合要求时可能返回原对象。
- **常见坑**：未接返回值，后续仍在用旧张量；改为float32不自动归一化；算错后再改类型不能恢复已经丢失的信息。
- **适用场景**：计算前选对数值类型，或把相互运算的张量送到同一可用设备。

### 教师核对来源

官方资料（按本机2.11版API核对）：
[1] https://docs.pytorch.org/docs/2.11/generated/torch.tensor.html
[2] https://docs.pytorch.org/docs/2.11/generated/torch.reshape.html
[3] https://docs.pytorch.org/docs/2.11/generated/torch.permute.html
[4] https://docs.pytorch.org/docs/2.11/generated/torch.unsqueeze.html
[5] https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.to.html
[6] https://docs.pytorch.org/docs/2.11/generated/torch.equal.html
[7] https://docs.pytorch.org/docs/2.11/generated/torch.cuda.is_available.html
[8] https://docs.pytorch.org/docs/2.11/tensor_attributes.html
[9] https://docs.pytorch.org/docs/2.11/notes/broadcasting.html
[10] https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

## 【完整代码示例 + 逐行注释】

只研究一个疑问：**“HWC改成CHW，只要shape相同，就能认为每个像素的颜色对应正确吗？”** 数据采用小张量，无图片文件、下载、模型或训练；批次轴和设备检查只用来确认转换后的输入状态。

```python
import torch  # 导入本课使用的张量计算库。
data = [[[11, 21, 31], [12, 22, 32]], [[13, 23, 33], [14, 24, 34]]]  # 行、列、RGB的嵌套顺序。
hwc = torch.tensor(data, dtype=torch.uint8, device='cpu')  # 函数①：明确按原始像素类型创建HWC张量。
floating = hwc.to(dtype=torch.float32)  # 函数⑩：只改数值类型，不自动缩放像素。
regrouped = torch.reshape(floating, (3, 2, 2))  # 函数④：把原有数字按目标形状重新分组。
chw = torch.permute(floating, (2, 0, 1))  # 函数⑤：新轴依次来自旧的通道、行、列轴。
bchw = torch.unsqueeze(chw, 0)  # 函数⑥：在最前面补上单张样本的批次轴。
print('创建/改类型：', hwc.dtype, floating.dtype, torch.equal(hwc, floating))  # 函数②⑦：另查dtype，比较数字是否保持。
print('重组/换轴shape：', regrouped.shape, chw.shape)  # 函数②：先看两种结果的形状。
print('重组/换轴全量相等：', torch.equal(regrouped, chw))  # 函数⑦：相同形状不能代替逐元素核对。
print('原像素/重组后/换轴后：', floating[0, 1, :], regrouped[:, 0, 1], chw[:, 0, 1])  # 函数②：追踪同一行列坐标的颜色分量。
print('批次shape/首张相等：', bchw.shape, torch.equal(bchw[0], chw))  # 函数②⑦：新增轴后原图数据仍对应。
device = torch.device('cpu')  # 函数⑨：默认选CPU，保证无CUDA也能学完。
if torch.cuda.is_available():  # 函数⑧：只有检测可用时才选择CUDA。
    device = torch.device('cuda:0')  # 函数⑨：描述实际存在的CUDA目标。
on_device = bchw.to(device=device)  # 函数⑩：接住设备转换结果，不改变原变量。
back_on_cpu = on_device.to(device='cpu')  # 函数⑩：回CPU后用一致设备核对数值。
print('原设备/目标设备/往返相等：', bchw.device, on_device.device, torch.equal(back_on_cpu, bchw))  # 函数②⑦：确认搬运后数字保持。
```

**真实运行结果**（学生固定解释器；从本文件第一段Python代码实跑注入，教师 `.venv` 因缺少torch未通过）：

```text
创建/改类型： torch.uint8 torch.float32 True
重组/换轴shape： torch.Size([3, 2, 2]) torch.Size([3, 2, 2])
重组/换轴全量相等： False
原像素/重组后/换轴后： tensor([12., 22., 32.]) tensor([21., 32., 14.]) tensor([12., 22., 32.])
批次shape/首张相等： torch.Size([1, 3, 2, 2]) True
原设备/目标设备/往返相等： cpu cuda:0 True
```

读法：先看“重组/换轴shape”和全量比较，再检查同一行列位置的颜色分量，最后核对批次轴与搬运后的值。类型转换前后相等这一行只说明数值相等，dtype要分别看；设备迁移也不会替你改轴或缩放。

**已声明的设备差异**：当前本机CUDA可用；其他机器可显示CPU目标，此时最后一行中间设备为 `cpu`，有CUDA时可能显示 `cuda:0`。这是设备条件差异，不能把布尔比较失败、dtype变化或像素对应错误算作版本噪声；本课其余输出没有已知显示豁免。

## 【课后练习】

### 一道迁移调查：通道偏移放在换轴前后，判断会变吗？

现有一个按HWC记录、颜色顺序为RGB的原始像素列表 `pixels`，以及按RGB顺序排列的通道偏移列表 `offsets`。偏移用于测试数值运算，要求每个像素的各颜色分量分别减去同色偏移；允许结果为负，不截断、不取绝对值、不缩放，也不改变输入RGB含义。输入列表和数值仅作实验材料，按原样使用：

- `pixels`：`[[[3, 22, 60], [9, 45, 80], [18, 65, 95]], [[25, 15, 120], [40, 35, 140], [55, 75, 160]]]`。
- `offsets`：`[8, 30, 70]`。

请先把原始像素和偏移材料都明确保存为 `torch.uint8` 张量，保留原输入。随后独立选择本课工具完成调查，不读取图片、不引入新库、不训练模型，计算可以全程在CPU完成。

**预测**：运行前写下你的判断：两个操作数都保持原整数类型，先相减再改浮点，是否和先把两者改成浮点再相减一样可靠？另一种说法认为“改成CHW后，原样保留按RGB排好的一维偏移表，程序只要能算，偏移就仍是逐通道的”；你是否认同？预测可被实际结果推翻，不需要先写数值答案。

**实跑**：提供能比较上述判断的实际证据，同时给出符合题意的带批次轴结果。分别调查计算所需类型和偏移实际对齐的轴，不把两种问题混在一个差异中。用至少一个真实像素的各颜色分量和整体比较支撑解释；不用输出每个中间变量，也不要求固定打印格式。

**解释**：根据实跑保留或修正预测，说明不带负号的整数为什么影响操作顺序、偏移参数怎样对应通道轴、为什么shape和“代码没报错”都不足以证明颜色含义正确。最后打印结果的shape、dtype与device；可只选CPU，也可检查CUDA后选择设备，说明接住返回值的必要性。

#### 验收点

- 原像素输入shape为 `(2, 3, 3)`，原偏移shape为 `(3,)`，两者初始dtype均为 `torch.uint8`；原输入内容保持原样。
- 正确结果shape为 `(1, 3, 2, 3)`，dtype为 `torch.float32`；device为CPU或当前检测可用的CUDA设备，搬回CPU后元素保持一致。
- 完整数据比较必须支持判断，不只给shape；预测允许出错，但实跑后须按证据修正，不能把原预测当作结论。
- 类型顺序实验使用相同输入和偏移；整数路径中两个减法操作数均为uint8，浮点路径中两者均为float32，区分“先计算再改浮点”与“先改浮点再计算”；最终正确结果保留应有的负值。
- 轴对齐实验在可表示负数的同一类型中进行；至少一次整体比较与一个像素的颜色对应证据共同支持解释。参数不能只因程序可运行就被认定为逐通道。
- 所有像素都满足原行列坐标对应、各颜色分量减去同色偏移的要求；不截断、不取绝对值、不把RGB倒置、不引入归一化。打印必要证据即可，无固定措辞或print数量要求。
- 本题未提供私有参考代码或正确像素答案；验收看真实计算与解释，不要求复刻演示的实现顺序。

> ⚠️ 固定使用你的 conda **BaiQing3.10AI**，解释器为 `D:/Conda_envs/envs/BaiQing3.10AI/python.exe`；提交完整代码、预测、真实输出和运行后的解释。练习代码不要求逐行注释。代码写入本课 `test.py`；当前练习待提交，课件和教师实跑不等于本人独立掌握。

### 我的练习解答
```python
import torch

pixels = [[[3, 22, 60], [9, 45, 80], [18, 65, 95]], [[25, 15, 120], [40, 35, 140], [55, 75, 160]]]
offsets = [8, 30, 70]

hwc_pixels = torch.tensor(pixels, dtype=torch.uint8, device='cpu')
floating = hwc_pixels.to(dtype=torch.float32)  # 函数⑩：只改数值类型，不自动缩放像素。
hwc_offsets = torch.tensor(offsets, dtype=torch.uint8, device='cpu')
# 预测：不认同，应该先相减再改浮点，偏移后位置就不对了，相减是对应的位置。
print(hwc_pixels.shape, hwc_pixels.dtype)
print(hwc_offsets.shape, hwc_offsets.dtype)

pixels_j = hwc_pixels - hwc_offsets
pixels_chw = torch.permute(floating, (2, 0, 1))
pixels_b_chw = torch.unsqueeze(pixels_chw, 0)
print(pixels_b_chw.shape, pixels_b_chw.dtype)
if torch.cuda.is_available():
	device = torch.device('cuda')
on_device = pixels_b_chw.to(device=device)  # 函数⑩：接住设备转换结果，不改变原变量。
back_on_cpu = on_device.to(device='cpu')  # 函数⑩：回CPU后用一致设备核对数值。
```
```
torch.Size([2, 3, 3]) torch.uint8
torch.Size([3]) torch.uint8

torch.Size([1, 3, 2, 3]) torch.float32
```
### 我的疑问


## 【本节重点总结】

1. shape只描述每根轴有多长，轴名与RGB含义要靠正确的索引和换轴关系维护。
2. reshape重新分组、permute重排轴、unsqueeze补单元素轴，类型转换与数值缩放也各有职责。
3. 先判断计算需要的dtype和参数对齐方向，再接住设备转换结果；CPU足以完成本课，不涉及训练。

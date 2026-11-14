# 第九课：PyTorch基础二：Dataset与DataLoader

> 2026-11-14开课。第八课已按用户授权由教师修正并实跑通过；修正版功能通过不等于本人独立掌握，下一步用数据管道变式继续验证。
> 学生继续使用 `D:/Conda_envs/envs/BaiQing3.10AI/python.exe`。本课教师演示与练习准备均在该解释器下验证；教师 `.venv` 缺少 `torch`，不改环境，也不声称双环境通过。

## 【本节核心知识点】

理解 Dataset 如何按索引提供一个样本，理解 DataLoader 如何把多个样本组成批次，并能坚持“训练集学习预处理统计量，验证集只使用已学统计量”的数据流规则。

## 【核心概念讲解】

### 1. Dataset 与 DataLoader 分工

Dataset（数据集）解决的是“第几个样本是什么”：给它一个索引，它返回一条样本记录，通常是 `(image, label)`。DataLoader（数据加载器）解决的是“怎样把样本送给模型”：它从 Dataset 中取样本，按 `batch_size` 组成批次，并可在训练时打乱顺序。

可以把两者分成两层：

- Dataset 是单样本层：输入一个 `index`，输出一张图像和一个标签。
- DataLoader 是批次层：输入一个 Dataset，输出一批图像和一批标签。

对于图像分类，单张图像通常是 CHW；经过 DataLoader 组成批次后，图像通常是 BCHW。B 是 batch（批次）轴，不是图像本身新增的颜色或空间轴。

### 2. 自定义 Dataset 的三个关键接口

自定义数据集通常继承 `torch.utils.data.Dataset`，并实现三个部分：

1. `__init__`：保存图像、标签和预处理配置；它负责准备对象，不负责根据索引返回样本。
2. `__len__`：返回数据集有多少条样本；DataLoader 会据此判断索引范围和批次数。
3. `__getitem__`：接收一个索引，取出对应图像和标签，并执行这条样本需要的转换。

`__getitem__` 的返回值必须与标签保持对应。不能只对图像排序或筛选而忘记同步处理标签，否则程序可能正常运行，但图像与标签已经错配。

### 3. 训练集与验证集的预处理边界

归一化可以写成：

```text
标准化结果 = (原值 - 均值) / 标准差
```

关键不只是公式，而是均值和标准差从哪里来：

- 训练集可以计算均值和标准差，因为训练集负责学习预处理规则。
- 验证集、测试集不能重新计算自己的规则；它们只使用训练集已经得到的均值和标准差。

如果验证集也参与计算均值和标准差，验证数据的信息就提前进入了预处理规则，形成数据泄漏。Dataset 可以保存训练集统计量，再在 `__getitem__` 中应用到训练样本和验证样本。

### 4. DataLoader 的批次行为

`batch_size` 决定一批最多取多少条样本；`shuffle=True` 通常用于训练集，每轮重新打乱索引；验证集通常使用 `shuffle=False`，让检查顺序稳定；`drop_last=True` 会丢弃最后不足一个完整批次的样本，`drop_last=False` 则保留最后的不完整批次。

因此，DataLoader 的长度不是 Dataset 的长度，而是“按批次切分后有几批”。读取结果的第一根轴是 B，标签也会从一个标量变成一批标签。

### 5. 本课演示的观察顺序

本课演示只保留一个核心链路：

> 原始张量 → 训练/验证切分 → 训练统计量 → Dataset 单样本 → DataLoader 批次

先直接读取一条样本，确认 Dataset 的返回值；再读取训练批次，观察 BCHW 和标签批次；最后读取验证批次，确认验证集复用训练统计量且不打乱顺序。

## 【核心内置函数拆解】

本节的“内置函数”栏目包含 PyTorch 库函数、类、方法和 Python 特殊方法。语法中的 `x` 表示已有张量，`dataset` 表示已有数据集对象。

### ① `torch.manual_seed`：固定随机数顺序

- **作用**：设置 PyTorch 随机数生成器的种子，使本次进程中的随机操作更容易复现。
- **完整语法**：`torch.manual_seed(seed)`。
- **关键参数**：`seed` 是整数种子；固定种子不代表所有设备、所有并行设置下都自动完全一致，本课使用单进程 CPU/CUDA 条件下的简单批次观察。
- **返回值**：返回一个 PyTorch 随机数生成器对象；本课只关心它对后续随机顺序的影响。
- **常见坑**：只设置种子不能把 `shuffle=False` 变成随机；每次重新创建 DataLoader、随机生成器和调用顺序也会影响结果。
- **适用场景**：复查 `shuffle=True` 的样本顺序，或制作可重复的最小实验。

### ② `torch.tensor`：创建图像与标签张量

- **作用**：根据列表等已有数据创建 Tensor。
- **完整语法**：`torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False)`。
- **关键参数**：图像数据本课使用 `dtype=torch.float32`；分类标签本课使用 `dtype=torch.int64`；图像和标签的第一根轴必须一一对应。
- **返回值**：一个 Tensor。
- **常见坑**：图像与标签数量不对应；把标签误建成浮点类型；把样本轴、通道轴和空间轴的顺序弄混。
- **适用场景**：把小型内存数据整理成 Dataset 可以读取的输入。

### ③ `Tensor.mean`：计算均值

- **作用**：计算张量元素的平均值；本课用训练图像计算归一化中心。
- **完整语法**：`x.mean(dim=None, keepdim=False, *, dtype=None)`；本课使用无参数形式计算训练张量全部元素的均值。
- **关键参数**：`dim` 指定沿哪根轴计算；`keepdim` 是否保留被计算的轴；本课不把图像逐通道统计扩展为新问题。
- **返回值**：一个 Tensor 标量或按指定轴保留维度的 Tensor。
- **常见坑**：把验证集也放入统计量计算；在错误轴上计算导致每个通道、每个像素得到不同形状的统计量。
- **适用场景**：从训练数据学习标准化所需的均值。

### ④ `Tensor.std`：计算标准差

- **作用**：计算张量元素的离散程度；本课用训练图像计算归一化尺度。
- **完整语法**：`x.std(dim=None, *, correction=1, keepdim=False)`；本课使用 `x.std(unbiased=False)` 的兼容写法计算总体标准差。
- **关键参数**：`unbiased` 或新版本等价参数控制样本标准差与总体标准差的区别；本课固定使用总体标准差，避免小数据实验的统计口径变化。
- **返回值**：一个 Tensor 标量或按指定轴保留维度的 Tensor。
- **常见坑**：训练集和验证集分别计算；小数据下把不同统计口径混用；标准差为零时直接除法会产生无效数值。
- **适用场景**：从训练数据学习标准化所需的尺度。

### ⑤ `torch.utils.data.Dataset`：自定义数据集基类

- **作用**：提供自定义数据集的接口约定，让对象能够按索引提供样本。
- **完整语法**：`class MyDataset(torch.utils.data.Dataset)`，并实现 `__len__(self)` 与 `__getitem__(self, index)`。
- **关键参数**：继承关系决定接口；`__init__` 保存原始数据和转换规则；`__getitem__` 的 `index` 决定取哪条样本。
- **返回值**：基类本身不直接返回样本；具体子类的 `__getitem__` 返回一条样本记录。
- **常见坑**：漏写 `__len__`；图像和标签索引不同步；在构造对象时把验证集统计量重新 fit。
- **适用场景**：内存数组、图片路径、表格记录等需要统一按索引读取的场景。

### ⑥ `__len__`：报告数据集长度

- **作用**：告诉 Python 和 DataLoader 当前数据集有多少条样本。
- **完整语法**：`def __len__(self):`，函数体返回整数。
- **关键参数**：返回值必须是非负整数，并与可接受的索引范围一致。
- **返回值**：Python 整数；调用 `len(dataset)` 时由 Python 读取。
- **常见坑**：返回图像的通道数、像素数或标签长度以外的错误数量；切分图像后忘记同步切分标签。
- **适用场景**：检查样本数量、计算 DataLoader 批次数和验证索引范围。

### ⑦ `__getitem__`：按索引返回一条样本

- **作用**：根据一个索引取出一条图像和对应标签，并可在取出时执行预处理。
- **完整语法**：`def __getitem__(self, index):`，函数体返回一条样本记录。
- **关键参数**：`index` 是样本位置；本课返回 `(image, label)`；归一化使用训练集保存的 `mean` 和 `std`。
- **返回值**：一条样本，通常是图像 Tensor 与标签 Tensor 的二元组。
- **常见坑**：只变换图像不变换标签；返回的数据类型不适合后续损失函数；把 batch 维提前塞进单样本。
- **适用场景**：实现图片读取、张量转换、归一化、数据增强等单样本处理。

### ⑧ `DataLoader`：批量读取数据集

- **作用**：从 Dataset 中按批次取样本，处理批次组合、顺序和是否丢弃尾批次。
- **完整语法**：`DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, *, prefetch_factor=None, persistent_workers=False, pin_memory_device='')`；本课只使用前面几个常用参数。
- **关键参数**：`dataset` 是数据来源；`batch_size` 是批次大小；`shuffle` 控制是否打乱索引；`drop_last` 控制是否丢弃尾批次；`num_workers=0` 表示主进程读取，便于本课观察。
- **返回值**：一个可迭代的数据加载器；每次迭代得到一批图像和一批标签。
- **常见坑**：把 Dataset 长度当成 DataLoader 长度；验证集不必要地打乱；`drop_last=True` 丢掉尾部样本却没有意识到；多进程读取增加调试复杂度。
- **适用场景**：训练循环、验证循环和需要批量送入模型的任务。

### 教师核对来源

- https://docs.pytorch.org/docs/stable/data.html
- https://docs.pytorch.org/docs/stable/generated/torch.utils.data.Dataset.html
- https://docs.pytorch.org/docs/stable/generated/torch.utils.data.DataLoader.html
- https://docs.pytorch.org/docs/stable/generated/torch.manual_seed.html

## 【完整代码示例 + 逐行注释】

只研究一个疑问：**“一个 Dataset 样本怎样变成训练批次，同时怎样守住训练集统计量边界？”** 数据使用内存中的小张量，不读取图片文件、不下载数据、不训练模型。

```python
import torch  # 导入张量计算库。
from torch.utils.data import Dataset, DataLoader  # 导入数据集基类和批量加载器。

torch.manual_seed(7)  # 函数①：固定随机顺序，便于复查shuffle=True的批次。
images = torch.tensor([  # 函数②：创建5张单通道、2×2的小图像。
    [[[0.0, 1.0], [2.0, 3.0]]],  # 第0张图像。
    [[[4.0, 5.0], [6.0, 7.0]]],  # 第1张图像。
    [[[8.0, 9.0], [10.0, 11.0]]],  # 第2张图像。
    [[[12.0, 13.0], [14.0, 15.0]]],  # 第3张图像。
    [[[16.0, 17.0], [18.0, 19.0]]],  # 第4张图像。
], dtype=torch.float32)  # 明确使用浮点类型，便于后续归一化。
labels = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int64)  # 函数②：创建与图像一一对应的类别标签。

train_images = images[:4]  # 切出训练图像；训练集负责学习归一化统计量。
train_labels = labels[:4]  # 切出训练标签。
val_images = images[4:]  # 切出验证图像；验证集只使用训练集统计量。
val_labels = labels[4:]  # 切出验证标签。
train_mean = train_images.mean()  # 函数③：只用训练图像计算均值。
train_std = train_images.std(unbiased=False)  # 函数④：只用训练图像计算总体标准差。

class SmallImageDataset(Dataset):  # 函数⑤：定义自己的图像数据集类。
    def __init__(self, images, labels, mean, std, normalize=True):  # 函数⑤：保存数据和预处理配置。
        self.images = images  # 保存图像张量。
        self.labels = labels  # 保存标签张量。
        self.mean = mean  # 保存训练集均值。
        self.std = std  # 保存训练集标准差。
        self.normalize = normalize  # 保存是否执行归一化的开关。

    def __len__(self):  # 函数⑥：返回数据集中的样本数量。
        return self.images.shape[0]  # 第一根轴是样本轴。

    def __getitem__(self, index):  # 函数⑦：按一个索引返回一份样本和标签。
        image = self.images[index]  # 取出一张图像，保留C、H、W三根轴。
        label = self.labels[index]  # 取出该图像对应的标签。
        if self.normalize:  # 根据配置决定是否执行归一化。
            image = (image - self.mean) / self.std  # 使用训练集统计量处理当前图像。
        return image, label  # 返回Dataset的一条样本记录。

raw_dataset = SmallImageDataset(images, labels, train_mean, train_std, normalize=False)  # 函数⑤：保留一个不归一化的数据集作对照。
train_dataset = SmallImageDataset(train_images, train_labels, train_mean, train_std, normalize=True)  # 函数⑤：建立训练数据集。
val_dataset = SmallImageDataset(val_images, val_labels, train_mean, train_std, normalize=True)  # 函数⑤：验证集复用训练统计量。
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False, num_workers=0)  # 函数⑧：把训练样本组成批次并打乱顺序。
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)  # 函数⑧：验证集按固定顺序逐样本读取。

raw_image, raw_label = raw_dataset[0]  # 函数⑦：直接读取一个未归一化样本。
normalized_image, normalized_label = train_dataset[0]  # 函数⑦：直接读取一个归一化样本。
print('单样本：', raw_image.shape, raw_image.dtype, raw_label, normalized_label)  # 函数②⑦：观察一条样本的图像形状、类型和标签。
print('归一化首样本首像素：', raw_image[0, 0, 0], normalized_image[0, 0, 0])  # 函数⑦：比较归一化前后的同一像素。
print('Dataset长度：', len(train_dataset), len(val_dataset))  # 函数⑥：确认训练集和验证集的样本数量。

for batch_images, batch_labels in train_loader:  # 函数⑧：从DataLoader逐批取出训练数据。
    print('训练批次：', batch_images.shape, batch_images.dtype, batch_labels.tolist())  # 函数⑧：观察BCHW批次和标签批次。
    break  # 只观察第一个批次，不重复打印全部数据。

for batch_images, batch_labels in val_loader:  # 函数⑧：从验证DataLoader读取验证数据。
    print('验证批次：', batch_images.shape, batch_labels.tolist())  # 函数⑧：确认验证集批次顺序和形状。
    break  # 只观察第一个验证批次。
print('Loader批次数：', len(train_loader), len(val_loader))  # 函数⑧：确认batch_size和drop_last共同决定批次数。
```

**真实运行结果**（学生固定解释器 `BaiQing3.10AI`；教师 `.venv` 因缺少 `torch` 未通过）：

```text
单样本： torch.Size([1, 2, 2]) torch.float32 tensor(0) tensor(0)
归一化首样本首像素： tensor(0.) tensor(-1.6270)
Dataset长度： 4 1
训练批次： torch.Size([2, 1, 2, 2]) torch.float32 [1, 0]
验证批次： torch.Size([1, 1, 2, 2]) [0]
Loader批次数： 2 1
```

读法：单样本仍是 CHW，送进 DataLoader 后多出最前面的 B 轴成为 BCHW；训练批次的标签也从一个标量变成一批标签。训练集均值和标准差在 Dataset 创建前只由训练图像计算，验证 Dataset 直接复用这两个统计量。

## 【课后练习】

### 一道迁移练习：索引切分与尾批次判断

使用下面给出的 `samples` 和 `labels`，独立写一个继承 `Dataset` 的数据集类，并用 `DataLoader` 调查“自定义索引切分后，`drop_last` 会怎样改变训练批次”。本题不读取图片、不下载数据、不引入新库、不训练模型。

```python
samples = [
    [[[1.0, 2.0], [3.0, 4.0]]],
    [[[5.0, 6.0], [7.0, 8.0]]],
    [[[9.0, 10.0], [11.0, 12.0]]],
    [[[13.0, 14.0], [15.0, 16.0]]],
    [[[17.0, 18.0], [19.0, 20.0]]],
    [[[21.0, 22.0], [23.0, 24.0]]],
]
labels = [1, 0, 1, 0, 1, 0]
```

请完成以下任务：

1. 将列表材料转换为图像张量和标签张量，并保持图像与标签的对应关系。
2. 选择不连续的训练索引和验证索引，建立两个 Dataset；训练集和验证集不能通过重新计算各自统计量来归一化。
3. 在训练 Dataset 中实现单样本读取；验证 Dataset 必须复用训练集学到的均值和标准差。
4. 建立训练 DataLoader 和验证 DataLoader；训练加载器使用打乱，验证加载器保持顺序，并让训练加载器丢弃不完整尾批次。
5. 运行前预测：训练集最后一条样本是否会出现在第一批或任何批次中；验证批次的标签顺序是否保持索引顺序。再用实跑结果检查预测。
6. 输出一个训练样本和一个验证样本的图像形状、标签与归一化后像素；输出训练批次和验证批次的图像形状、标签形状、dtype 与标签内容；最后解释 Dataset 的单样本层和 DataLoader 的批次层分别做了什么。

#### 验收点

- 图像张量为 `torch.float32`，标签张量为 `torch.int64`；图像和标签按原始对应关系保存。
- 训练/验证 Dataset 的长度分别为 `4` 和 `2`；自定义 `__getitem__` 返回单张 CHW 图像和对应标签，未把批次轴提前塞入单样本。
- 训练统计量只由训练索引对应的样本计算；验证样本使用同一组训练统计量，不重新计算验证统计量。训练均值应为 `12.5`；训练标准差使用总体标准差口径。
- 训练 DataLoader 使用 `batch_size=3`、`shuffle=True`、`drop_last=True`、`num_workers=0`；验证 DataLoader 使用 `batch_size=2`、`shuffle=False`、`drop_last=False`、`num_workers=0`。
- 训练 DataLoader 批次数为 `1`，验证 DataLoader 批次数为 `1`；训练批次图像形状为 `(3, 1, 2, 2)`，验证批次图像形状为 `(2, 1, 2, 2)`。
- 训练批次标签数量为 `3`，验证批次标签按验证索引顺序保留 `2` 个标签；训练被丢弃的尾部样本不能出现在训练批次中。
- 至少一次完整实跑证据同时包含：单样本 CHW、批次 BCHW、标签批次、dtype、训练统计量边界和预测复核；不能只打印 shape。
- 解释必须区分 Dataset 与 DataLoader：前者按索引提供一条样本，后者按批次组织样本；必须说明 `drop_last=True` 的实际影响。

> ⚠️ 固定使用你的 conda **BaiQing3.10AI**，解释器为 `D:/Conda_envs/envs/BaiQing3.10AI/python.exe`；提交完整代码、预测、真实输出和运行后的解释。练习代码不要求逐行注释。代码写入本课 `test.py`；当前练习尚未提交，不把教师演示和参考答案当作本人独立掌握。

### 我的练习解答


### 我的疑问


## 【本节重点总结】

1. Dataset 负责按索引返回一条样本，DataLoader 负责把多条样本组成批次。
2. 训练集可以学习归一化统计量，验证集和测试集只使用训练集统计量，不能重新 fit。
3. 单样本通常是 CHW，批次通常是 BCHW；`batch_size`、`shuffle` 和 `drop_last` 会共同决定读取方式和批次数。

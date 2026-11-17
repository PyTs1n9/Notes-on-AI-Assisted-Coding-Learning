# 补充答疑：梯度、优化器、模型与 Normalize

> 2026-11-17，回应本人提出的四个遗留问题。对应第10课训练循环、第11课模型权重，以及图像增强阶段的输入标准化。下面是教师解释和教师实跑证据，不代表本人已独立掌握，也不改变现有课程验收状态。

先用一个问题定位你卡在哪里：**只执行 `loss.backward()`，没有执行 `optimizer.step()`，模型参数会不会改变？** 可以先记下判断，再对照第一节；不需要先答对才能阅读答案。

## 1. 梯度是什么？优化器是什么？两者如何合作？

模型的权重、偏置是可调整的数字，统称“参数”。损失是衡量当前预测与目标差距的数字，是训练希望减小的量。

**梯度 gradient（梯度）描述：在当前参数位置，参数发生微小变化时，损失会怎样变化。** 对一个参数来说，它就是损失对该参数的导数；多个参数的这些导数组合起来，就是梯度。导数可以先理解成“当前位置的变化率”。

只看某一个参数、暂时保持其他参数不变：

- 梯度为正：参数稍微增大，损失倾向于增大；希望减小损失，通常应向参数减小的方向走。
- 梯度为负：参数稍微增大，损失倾向于减小；通常应向参数增大的方向走。
- 梯度绝对值较大：损失在当前位置对这个参数更敏感。它不等于“这个参数更重要”，也不意味着可以任意走大步。
- 梯度为零：当前位置的一阶变化率为零，不能据此断定已经找到最小损失。

**梯度指向局部上升最快的方向，所以基础梯度下降沿着负梯度调整参数。** “局部”表示只描述当前位置附近；学习率过大可能跨过合适位置，使更新后的损失反而变大。

**优化器 optimizer（优化器）负责按照指定规则更新参数。** 它接收需要管理的模型参数，使用这些参数的 `.grad`，结合学习率和算法状态决定怎么改。学习率是控制调整步幅的设置。

最基础、无动量且无权重衰减的 SGD，其规则是：

$\text{新参数}=\text{旧参数}-\text{学习率}\times\text{梯度}$

这条规则用于理解方向，不要求手算。Adam 也使用梯度，但会结合历史梯度信息分别调整各参数的更新步幅，不应把它理解成永远只做上面这一条计算。

| 步骤 | 接收什么 | 做什么 | 得到什么 |
|---|---|---|---|
| 前向 | 输入数据、当前参数 | 按模型结构计算 | 预测分数 |
| 计算损失 | 预测分数、真实标签 | 衡量预测差距 | 损失 |
| `loss.backward()` | 损失及其计算图 | 自动求导 | 各参数的 `.grad` |
| `optimizer.step()` | 优化器管理的参数及其 `.grad` | 按更新规则修改参数 | 新参数 |

因此，当前普通训练代码中，**只执行 `backward` 会产生梯度，不会更新模型参数；`step` 才负责更新参数。** 优化器并不是在普通 `step()` 调用中自行重新计算本次损失和梯度。

## 2. 梯度清零能放在前向传播之前吗？为什么？

**可以。当前课程中，每个批次独立更新时，放在前向之前很常见。**

先区分参数本身和参数的 `.grad`：参数保存当前权重；`.grad` 保存反向传播算出的梯度。`zero_grad` 清的是后者，不会把模型权重变成零，也不会删除模型结构。

普通前向使用输入和权重计算结果，不读取参数的 `.grad`。所以把清理旧梯度提前到前向之前，不会改变本轮前向结果。反向传播则默认往 `.grad` 里累加，清理旧值是为了让本轮反向得到本轮所需的梯度。

两种常规顺序都成立：

- 清旧梯度 → 前向 → 算损失 → 反向 → 更新参数。
- 前向 → 算损失 → 清旧梯度 → 反向 → 更新参数。

**关键是：开始本轮梯度计算前，没有混入不想保留的旧梯度；更新时，本轮要使用的梯度还在。** 这条原则比死记某一行的位置更可靠。若在上一轮 `step` 后立刻清理，直到下一轮反向前也没有别的反向操作，同样可以。

反向 → 清梯度 → 更新，这个顺序会清掉刚算出的更新依据。在本笔记明确使用 `set_to_none=True` 时，梯度变成 `None`，SGD 跳过对应参数。不要推广成“梯度张量全为零时，所有优化器都绝对不会更新参数”：动量等历史状态、权重衰减可能使行为不同。

另外，`step()` **不会自动清梯度**；`zero_grad()` 也不会重置 Adam 等优化器的历史状态。

例外用途是“梯度累积”：有意让多个小批次的梯度累加，再统一更新。这时不能每个小批次都清零；这是另一种训练安排，当前课程先按每批独立更新理解。

## 3. 手写 `nn.Module` 和调用 `torchvision.models` 有什么区别？

**两者创建出来的都可以是 PyTorch 模型；TorchVision 已经替你写好了常见网络结构。**

你在第11课 `practice.py` 中定义的 `SmallClassifier`，自己指定了隐藏层、激活函数、输出层，并在 `forward` 中决定它们如何连接。`models.resnet18` 则创建现成的 ResNet18 结构；ResNet 可以先理解成一种常用的图像神经网络。

| 比较项 | 自己继承 `nn.Module` | 调用 `torchvision.models` |
|---|---|---|
| 谁设计连接方式 | 你在初始化和 `forward` 中定义 | 库作者已实现既定网络 |
| 是否为 `nn.Module` 对象 | 是 | ResNet18 等同样是 |
| 是否天然训练好了 | 普通层默认随机初始化，除非自己加载权重 | 取决于是否指定预训练权重 |
| 能否使用自动求导和优化器 | 可以 | 可以 |
| 能否改结构 | 自己设计，较灵活 | 可替换输出层等，也可进一步改造 |
| 主要用途 | 学习原理、实现自定义网络 | 使用成熟结构、开展实际图像任务 |

必须分开“结构”和“权重”：**结构是计算路线，权重是这条路线中可调整的数值。** 调用现成结构，不代表已经获得训练成果。

- `models.resnet18(weights=None)`：创建现成结构，默认随机初始化，不下载预训练权重。
- 指定该模型的预训练权重枚举：创建结构并加载已经训练得到的权重；首次使用通常涉及下载，本次没有执行下载。

预训练也不等于自动适合你自己的类别。后续通常还要匹配输入预处理、调整分类输出层，再训练或微调。两种创建方式都要继续安排数据、损失、训练/验证流程；使用预训练模型做推理时也需要切换到评估模式。

## 4. Normalize 是归一化还是标准化？

你写的 `nomarlize` 应是指 `normalize`；实际查接口时要核对正确拼写。**仅凭这个英文名字无法判断计算方式，必须看来自哪个库、哪个接口。**

在本课程常用的区别里：

| 操作 | 计算方式 | 主要效果 | 常见接口 |
|---|---|---|---|
| 最小—最大归一化 | $(x-x_{min})/(x_{max}-x_{min})$ | 将参考数据缩放到指定区间，常见为 `[0,1]` | `sklearn.preprocessing.MinMaxScaler` |
| 标准化 | $(x-\mu)/\sigma$ | 按均值平移，再按标准差缩放 | `StandardScaler`；`torchvision.transforms.Normalize` 使用这一形式 |
| 向量归一化 | 向量除以其长度（范数） | 调整向量的整体长度，例如变成单位长度 | `torch.nn.functional.normalize` |

均值描述数据的中心，标准差描述围绕中心的分散程度。“归一化”有时被广义使用，文档翻译也未必严格区分，所以**看公式比看中文译名更可靠**。最小—最大公式的分母为零时需要专门处理；使用训练范围缩放新数据时，新数据也可能落到 `[0,1]` 之外。

**若你问的是图像课的 `torchvision.transforms.Normalize`：它按通道执行 `(像素值 - 给定均值) / 给定标准差`，属于标准化形式，不是把像素自动压到 `[0,1]`。**

它有四个很容易混淆的细节：

1. `mean` 和 `std` 是你传进去的数字，它不会自动对每张输入图计算统计量。RGB 图像通常分别提供 R、G、B 三组统计量。
2. 输出允许有负数，也不保证在 `[0,1]` 内。
3. 只有统计量与被统计的数据及统计口径相匹配，才会得到相应的零均值、单位标准差；使用预训练模型给定的统计量，不保证每张图片都恰好均值0、标准差1。
4. 统计量必须与输入数值尺度一致。常见流程先把原始像素转成浮点数并缩放，再按预训练权重要求标准化；标准化这一步不会替你先除以255。自己估计统计量时只使用训练数据，验证/测试复用；使用预训练权重则优先遵循它规定的预处理。

`torch.nn.functional.normalize` 是另一个接口，常见用途是把向量除以其长度，不要把它和图像的 `transforms.Normalize` 混用。

## 核心函数速查：六字段卡片

### `loss.backward`

- 作用：沿计算图计算梯度，并累加到相关叶子张量的 `.grad`。
- 完整语法：`Tensor.backward(gradient=None, retain_graph=None, create_graph=False, inputs=None)`。
- 关键参数：当前标量损失直接调用；非标量输出通常需要指定外部梯度，不在本次展开。
- 返回值：`None`；梯度通过 `.grad` 读取。
- 常见坑：把计算梯度当成更新参数；对同一张已经释放的计算图重复反向。
- 场景：训练中计算参数更新所需的梯度。

### `torch.optim.SGD`

- 作用：创建采用随机梯度下降规则的优化器。
- 完整语法：`torch.optim.SGD(params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False, foreach=None, differentiable=False, fused=None)`。
- 关键参数：`params` 是要管理的参数；`lr` 是学习率；`momentum` 是动量；`weight_decay` 是权重衰减。本例只改变学习率，其余保持基础行为。
- 返回值：SGD 优化器对象。
- 常见坑：优化器管理的参数与实际训练模型不是同一组参数。
- 场景：按梯度更新模型参数；本例用最基础配置观察方向。

### `optimizer.zero_grad`

- 作用：清理优化器管理参数的旧梯度。
- 完整语法：`optimizer.zero_grad(set_to_none=True)`。
- 关键参数：`set_to_none=True` 将梯度设为 `None`；`False` 将已有梯度张量清为零。
- 返回值：`None`。
- 常见坑：以为清的是权重；在需要使用本轮梯度的 `step` 之前把它清掉。
- 场景：每批独立更新时，为下一次梯度计算清理累积值。

### `optimizer.step`

- 作用：按优化器规则更新它管理的参数。
- 完整语法：本次 SGD 对象为 `optimizer.step(closure=None)`。
- 关键参数：当前普通训练不传 `closure`；该可选参数是重新计算损失的函数，本次不使用。
- 返回值：不提供 `closure` 的普通调用返回 `None`。
- 常见坑：以为它会自动反向或自动清梯度。
- 场景：`backward` 之后实施本次参数更新。

### `torchvision.models.resnet18`

- 作用：创建 ResNet18 模型。
- 完整语法：`torchvision.models.resnet18(*, weights=None, progress=True, **kwargs)`。
- 关键参数：`weights=None` 不加载预训练权重；`weights` 可指定对应权重枚举；`**kwargs` 是继续交给模型构造器的可选设置，例如无预训练权重时可设置 `num_classes`。
- 返回值：继承 `nn.Module` 的 ResNet 对象。
- 常见坑：混淆模型结构和预训练权重；预训练输出类别数量不一定匹配自己的任务。
- 场景：创建成熟图像分类网络。

### `torchvision.transforms.Normalize`

- 作用：按通道进行减均值、除标准差的变换。
- 完整语法：先创建 `transform = torchvision.transforms.Normalize(mean, std, inplace=False)`，再调用 `output = transform(tensor)`。
- 关键参数：`mean`、`std` 为每通道统计量序列，标准差不能为零；输入应为浮点图像张量；`inplace=False` 默认不直接修改输入。
- 返回值：构造时得到变换对象；调用对象时得到变换后的张量。
- 常见坑：输入PIL图片；把0～255像素与0～1尺度统计量混用；以为输出一定处于 `[0,1]`。
- 场景：图像送入模型前，匹配训练约定或预训练权重要求的输入尺度。

## 最小运行演示

代码分成三个观察部分：参数与梯度、两种清零位置、现成模型与图像标准化。仅为解释四个问题，不是第11课练习答案；没有下载模型或修改环境。

```python
import torch  # 导入张量与自动求导工具。
from torch import nn  # 导入神经网络模块。
from torchvision import models, transforms  # 导入现成网络和图像变换。
w = nn.Parameter(torch.tensor(0.0))  # 创建一个可学习参数。
optimizer = torch.optim.SGD([w], lr=0.1)  # 只管理w，使用基础SGD规则。
loss = (w - 3.0).square()  # 定义一个简单损失以观察参数更新。
loss.backward()  # 计算梯度；此处不会修改w本身。
print('before step:', 'w =', w.item(), 'grad =', w.grad.item())  # 查看反向后的参数与梯度。
optimizer.step()  # 使用已有梯度更新w。
print('after step:', 'w =', w.item(), 'grad =', w.grad.item())  # 查看更新后仍然保留的梯度。
optimizer.zero_grad(set_to_none=True)  # 将梯度清为None，不清除参数值。
print('after zero_grad:', 'w =', w.item(), 'grad =', w.grad)  # 区分参数与梯度的变化。
results = []  # 保存两种清零位置训练后的参数。
for position in ['before_forward', 'before_backward']:  # 比较前向前与反向前两个位置。
    torch.manual_seed(7)  # 每条路径重新设置同一种子，保证相同初始参数。
    model = nn.Linear(1, 1)  # 每条路径创建独立模型。
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 每条路径创建独立优化器。
    for epoch in range(2):  # 连续更新，以覆盖第二轮已有历史梯度的情况。
        if position == 'before_forward':  # 第一种位置在前向之前清理。
            optimizer.zero_grad(set_to_none=True)  # 清理上一轮梯度。
        prediction = model(torch.tensor([[1.0]]))  # 使用输入和当前参数计算预测。
        loss = (prediction - 2.0).square().mean()  # 计算当前损失。
        if position == 'before_backward':  # 第二种位置在损失之后、反向之前清理。
            optimizer.zero_grad(set_to_none=True)  # 同样在本次反向前清掉旧梯度。
        loss.backward()  # 计算当前梯度。
        optimizer.step()  # 按当前梯度更新参数。
    results.append(model.weight.detach().clone())  # 保存权重快照，不保留计算图。
    results.append(model.bias.detach().clone())  # 保存偏置快照。
print('same final parameters:', torch.equal(results[0], results[2]) and torch.equal(results[1], results[3]))  # 同时比较完整权重与偏置。
network = models.resnet18(weights=None)  # 创建现成结构，不下载预训练权重。
print('ResNet is nn.Module:', isinstance(network, nn.Module))  # 检查现成模型与基类的关系。
print('ResNet training mode:', network.training)  # 创建模型后不会自动进入评估模式。
image = torch.tensor([[[0.0, 0.5, 1.0]]])  # 准备单通道浮点图像。
transform = transforms.Normalize(mean=[0.5], std=[0.25])  # 指定统计量，不自动估计输入统计量。
normalized = transform(image)  # 执行减均值、除标准差。
print('Normalize output:', normalized)  # 直接观察输出是否局限在0到1。
print('input unchanged:', torch.equal(image, torch.tensor([[[0.0, 0.5, 1.0]]])))  # 检查默认操作未改输入。
```

### 真实运行结果

以下内容由本文件自身代码在现有 `BaiQing3.10AI` 的 CPU 上执行后捕获。教师课程venv缺torch，未改环境，不声称双环境通过。浮点末位显示误差不影响结论。

```text
before step: w = 0.0 grad = -6.0
after step: w = 0.6000000238418579 grad = -6.0
after zero_grad: w = 0.6000000238418579 grad = None
same final parameters: True
ResNet is nn.Module: True
ResNet training mode: True
Normalize output: tensor([[[-2.,  0.,  2.]]])
input unchanged: True
```

## 一个可选理解检查

某段普通训练代码把清梯度放在每次参数更新之后，并且下一轮反向之前没有任何其他反向操作。你判断它是否一定错误？说明判断依据即可，不要求手算，也不需要重交已关闭的第10课作业。

## 本次要记住的四句话

1. 反向计算梯度，优化器使用梯度更新参数。
2. 清理的是梯度缓存；普通前向不依赖这个缓存，所以可提前清理。
3. 手写模型与现成模型共享 `nn.Module` 体系，主要区别在于谁实现结构，以及是否加载已有权重。
4. `Normalize` 需要看完整接口；TorchVision 图像变换采用减均值、除标准差的形式。

## 官方参考

- [Normalize：按通道减均值、除标准差](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html)
- [TorchVision 模型与预训练权重](https://docs.pytorch.org/vision/stable/models.html)
- [向量归一化的另一接口：functional.normalize](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html)

# 第十课：PyTorch训练循环与自动求导

> 本课使用 `D:/Conda_envs/envs/BaiQing3.10AI/python.exe` 实跑。课程 `.venv` 当前缺少 `torch`，因此本课只报告指定 Conda 环境验证结果。

## 【本节核心知识点】

掌握 `nn.Module` 模型、前向计算、交叉熵损失、优化器和自动求导的协作，并能按“前向 → 损失 → `zero_grad` → `backward` → `step`”顺序写出一个小型分类训练循环。

## 【核心概念讲解】

前情提要：第九课按索引返回CHW单样本，DataLoader组成BCHW批次；训练统计量不能从验证集重新学习。本课先用内存特征聚焦参数怎样变化，之后再把同样训练步骤接回加载器。



### 1. 从样本分数到参数更新

模型（model）是把输入变成输出的可学习函数。`nn.Module` 是 PyTorch 中模型的基类；继承它的对象可以保存参数，并通过 `forward` 定义前向计算。参数（parameter）是训练中会被调整的权重和偏置。

`Linear(输入特征数, 输出特征数)` 是全连接层：每条输入有多少个特征，第一项就写多少；它输出多少个分数，第二项就写多少。分类时，输出通常叫 logits（类别原始分数），它们还不是概率，也不需要先手动做 softmax 才能交给 `CrossEntropyLoss`。

ReLU（Rectified Linear Unit，修正线性单元）会把负数变成 0，正数保持不变。它是激活函数，用来增加模型表达能力；只有一层 `Linear` 的模型可以学习简单边界，多层线性层中间通常加入 ReLU。

损失函数（loss function）把预测与真实答案的差异压缩成一个数。`CrossEntropyLoss` 用于多分类：输入是每类的 logits，标签是从 0 开始的整数类别编号。它内部会处理 logits 的归一化与对数损失，本课采用整数类别标签；该接口也支持同形状的概率标签，但那是另一种用法，本课不展开。

### 2. 自动求导与计算图

自动求导（automatic differentiation）是 PyTorch 根据运算过程自动计算导数的机制。只要参与运算的张量或参数需要梯度，PyTorch 就会记录相关运算，形成计算图（computation graph）。`requires_grad=True` 表示这个张量需要记录梯度；模型参数默认具有这个属性。

`loss.backward()` 从损失沿计算图反向传播，把每个参数对损失的梯度写入参数的 `.grad`。梯度表示“参数向哪个方向调整，损失会怎样变化”。梯度默认会累加，所以每一轮更新前要用 `optimizer.zero_grad()` 清掉上一轮的梯度；否则本轮会混入历史梯度。

### 3. 五步训练顺序

每一轮训练的固定顺序是：

1. 前向：`logits = model(x)`，模型根据输入产生分数。
2. 损失：`loss = loss_fn(logits, y)`，比较分数与标签。
3. 清梯度：`optimizer.zero_grad()`，清理旧梯度。
4. 反向：`loss.backward()`，计算当前损失对应的梯度。
5. 更新：`optimizer.step()`，用梯度修改参数。

漏掉前向就没有本轮预测；漏掉损失就没有优化目标；漏掉 `zero_grad` 会让梯度跨轮累加；漏掉 `backward` 就没有新梯度；漏掉 `step` 则参数不会更新，模型通常学不到东西。`zero_grad` 必须在本轮 `backward` 之前，且不能放在 `backward` 与 `step` 之间把刚算出的梯度清掉。也可以在本轮前向之前清梯度；关键是反向前已清旧梯度、更新前保留本轮梯度。

### 4. SGD 与 Adam

SGD（Stochastic Gradient Descent，随机梯度下降）按照当前梯度和学习率更新参数，概念简单、可控，学习率选择很重要。Adam（Adaptive Moment Estimation，自适应矩估计）会结合梯度的一阶和二阶历史信息，为不同参数自适应调整步长，通常更容易在初期跑通，但仍需要合理的学习率和训练设置。

二者都是优化器，不是损失函数，也不会替代 `backward`。`backward` 负责算梯度，优化器的 `step` 负责用梯度改参数。

## 【核心内置函数拆解】

本课卡片中的 `model`、`x`、`y`、`loss` 和 `optimizer` 表示已经创建的对象。

### ① `nn.Module` — 定义可训练模型的基类
- **作用**：保存模型层与参数，并约定模型如何进行前向计算。
- **完整语法**：`class ModelName(nn.Module):`，类中定义 `__init__` 与 `forward`。
- **关键参数**：继承 `nn.Module`；在 `__init__` 中用 `super().__init__()`；在 `forward` 中连接各层。
- **返回值**：实例化后得到一个模型对象；调用模型时返回 `forward` 的结果。
- **常见坑**：忘记调用父类初始化；把计算写在类外；直接改普通列表中的层导致参数没有被注册。
- **适用场景**：需要组合多层、激活函数和自定义计算时。

### ② `nn.Linear` — 进行全连接变换
- **作用**：把输入特征映射为指定数量的输出特征。
- **完整语法**：`nn.Linear(in_features, out_features, bias=True)`。
- **关键参数**：`in_features` 必须等于输入最后一维；`out_features` 是输出最后一维；`bias` 控制是否使用偏置。
- **返回值**：一个可训练的线性层；调用后返回新的 Tensor。
- **常见坑**：输入特征数写错；把分类类别数写到输入位置；把 logits 误当成概率。
- **适用场景**：表格特征、展平后的图像特征和分类头。

### ③ `nn.ReLU` — 保留正值的激活函数
- **作用**：将负输入变成 0，保留正输入。
- **完整语法**：`nn.ReLU(inplace=False)`。
- **关键参数**：`inplace` 是否原地修改输入；初学时保持默认 `False` 更容易理解。
- **返回值**：经过 ReLU 的 Tensor。
- **常见坑**：把它当成损失函数；认为它会输出概率；原地修改可能影响需要保留的计算图。
- **适用场景**：多层网络的线性层之间。

### ④ `nn.CrossEntropyLoss` — 计算多分类损失
- **作用**：比较多分类 logits 与整数类别标签，返回损失。
- **完整语法**：`nn.CrossEntropyLoss(weight=None, reduction='mean')`，再调用 `loss_fn(logits, target)`。
- **关键参数**：输入形状通常是 `(样本数, 类别数)`；标签形状通常是 `(样本数,)` 且类型为 `torch.int64`。
- **返回值**：默认是一个标量损失 Tensor。
- **常见坑**：先手动 softmax；标签用浮点数或 one-hot；类别编号超出范围。
- **适用场景**：互斥的多分类任务。

### ⑤ `optimizer.zero_grad` — 清除旧梯度
- **作用**：把优化器管理参数上的旧梯度清除。
- **完整语法**：`optimizer.zero_grad(set_to_none=True)`。
- **关键参数**：`set_to_none` 是否把梯度设为 `None`；默认行为适合常规训练。
- **返回值**：通常返回 `None`。
- **常见坑**：忘记调用导致梯度累加；放在 `backward` 后导致刚得到的梯度被清掉。
- **适用场景**：每轮反向传播前。

### ⑥ `Tensor.backward` — 计算反向梯度
- **作用**：从损失出发沿计算图计算梯度。
- **完整语法**：`loss.backward(gradient=None, retain_graph=False, create_graph=False)`。
- **关键参数**：标量损失通常直接调用；`retain_graph` 只有需要重复使用计算图时才考虑。
- **返回值**：通常返回 `None`，梯度写入参数的 `.grad`。
- **常见坑**：在没有计算图的张量上调用；重复反向却没有保留图；误以为它会自动更新参数。
- **适用场景**：训练循环中计算当前损失对参数的梯度。

### ⑦ `optimizer.step` — 根据梯度更新参数
- **作用**：使用已经计算好的梯度和优化器规则修改参数。
- **完整语法**：`optimizer.step(closure=None)`。
- **关键参数**：普通训练直接调用；某些优化器才需要 `closure`。
- **返回值**：通常返回 `None`。
- **常见坑**：没有先 `backward`；在 `backward` 之后、`step` 之前清掉当前梯度；学习率过大导致训练不稳定。
- **适用场景**：每轮反向传播之后更新模型。

### ⑨ torch.optim.SGD

- **作用**：按梯度方向与学习率更新参数。
- **完整语法**：`torch.optim.SGD(params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False)`（此处列本课常用参数形式，高级可选项不展开）。
- **关键参数**：params来自model.parameters；lr是学习率；本课momentum保持默认。
- **返回值**：SGD优化器。
- **常见坑**：只创建优化器不会训练，lr过大可能不稳定。
- **适用场景**：用简单更新规则理解训练。

### ⑩ torch.optim.Adam

- **作用**：结合梯度历史为参数调节更新。
- **完整语法**：`torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)`（此处列本课常用参数形式，高级可选项不展开）。
- **关键参数**：params为参数；lr为基础学习率；betas控制历史平滑。
- **返回值**：Adam优化器。
- **常见坑**：仍需要清梯度、反向及step，不保证一定优于SGD。
- **适用场景**：后续模型训练的另一种优化器选择。

### ⑪ Module.parameters

- **作用**：提供模型已注册参数。
- **完整语法**：`model.parameters(recurse=True)`（此处列本课常用参数形式，高级可选项不展开）。
- **关键参数**：recurse控制是否包括子层。
- **返回值**：参数迭代器，可供优化器遍历。
- **常见坑**：把未注册普通对象误认为模型参数；本课层保存为self属性。
- **适用场景**：向优化器提供要更新的权重。

### ⑫ Tensor.item

- **作用**：提取单元素张量的数值。
- **完整语法**：`tensor.item()`（此处列本课常用参数形式，高级可选项不展开）。
- **关键参数**：无额外参数，张量必须只有一个元素。
- **返回值**：Python数字。
- **常见坑**：多元素张量报错；转换后数字不再保留计算图，不能用于backward。
- **适用场景**：打印标量loss。

### ⑬ Tensor.argmax

- **作用**：找最大值的位置。
- **完整语法**：`tensor.argmax(dim=None, keepdim=False)`（此处列本课常用参数形式，高级可选项不展开）。
- **关键参数**：dim指定比较轴；本课dim=1沿类别轴。
- **返回值**：int64索引张量。
- **常见坑**：不指定dim会对整体取一个索引；它不输出概率。
- **适用场景**：从每条样本logits得到类别编号。

## 【完整代码示例 + 逐行注释】

```python
import torch  # 导入张量与自动求导工具。
from torch import nn  # 导入神经网络层和损失函数。
torch.manual_seed(7)  # 固定初始化随机种子。
x = torch.tensor([[0.0, 0.2], [0.2, 0.0], [1.0, 1.2], [1.2, 1.0]], dtype=torch.float32)  # 创建每条含两个特征的输入。
y = torch.tensor([0, 0, 1, 1], dtype=torch.int64)  # 创建从0开始的整数标签。
class SmallClassifier(nn.Module):  # 函数①：继承模型基类。
    def __init__(self):  # 初始化模型层。
        super().__init__()  # 函数①：先初始化父类，使层能正确注册。
        self.hidden = nn.Linear(2, 4)  # 函数②：把两个输入特征映射为四个中间特征。
        self.activation = nn.ReLU()  # 函数③：加入非线性激活。
        self.output = nn.Linear(4, 2)  # 函数②：输出两个类别的原始分数。
    def forward(self, inputs):  # 函数①：规定一次前向计算的顺序。
        hidden = self.hidden(inputs)  # 函数②：计算中间特征。
        activated = self.activation(hidden)  # 函数③：把负值变成0。
        return self.output(activated)  # 函数②：返回类别原始分数。
model = SmallClassifier()  # 函数①：创建模型及其可训练参数。
loss_fn = nn.CrossEntropyLoss()  # 函数④：定义整数标签对应的交叉熵损失。
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)  # 函数⑨⑪：把模型参数交给优化器，学习率为更新步长。
for epoch in range(5):  # 完整遍历同一训练数据，每次称为一个epoch（轮次）。
    logits = model(x)  # 函数①：通过模型对象调用forward，建立本轮计算图。
    loss = loss_fn(logits, y)  # 函数④：得到更新前的标量训练损失。
    optimizer.zero_grad()  # 函数⑤：先清除历史梯度。
    loss.backward()  # 函数⑥：计算梯度；本行不更新参数。
    optimizer.step()  # 函数⑦：使用当前梯度修改参数。
    print('epoch', epoch + 1, 'loss', round(loss.item(), 4))  # 函数⑫：提取单元素损失数值以便打印。
predictions = model(x).argmax(dim=1)  # 函数⑬：每条样本选择分数最大类别，本课仅观察不再反向。
print('logits shape', logits.shape)  # 确认样本轴和类别轴。
print('predictions', predictions.tolist())  # 显示训练后的类别预测。
print('targets', y.tolist())  # 显示用于训练的标签；这不是独立测试成绩。
print('requires_grad', model.hidden.weight.requires_grad)  # 观察模型权重默认需要梯度。
```

**真实运行结果**（指定 Conda 环境实测；课程 `.venv` 缺少 `torch`）：

```text
epoch 1 loss 0.7927
epoch 2 loss 0.7314
epoch 3 loss 0.6941
epoch 4 loss 0.6628
epoch 5 loss 0.6319
logits shape torch.Size([4, 2])
predictions [0, 0, 1, 1]
targets [0, 0, 1, 1]
requires_grad True
```

## 【课后练习】

### 一道迁移练习：清梯度能放在反向之后吗？

沿用演示的数据、模型结构、初始化种子、损失与优化器设置，独立写一个对照调查。先预测：“把清梯度移到反向之后、参数更新之前，只要每轮仍调用清梯度、反向与更新，模型照样会学习”是否成立。

建立两条从相同初始化开始的训练路径：一条采用本课常规训练顺序，另一条采用上述候选顺序；两条路径使用相同训练数据和轮数。分别记录训练损失，比较训练开始与结束时同一批输入的完整原始分数，再用实跑解释哪一步产生梯度、哪一步消耗梯度、错误放置清理会产生什么后果。只改顺序，不改模型或优化器来掩盖现象。

#### 验收点

- 输入为演示的二维特征及整数标签，模型结构及学习率与演示相同，种子7；每条路径重新设置种子并建立独立模型/优化器，共训练5轮。
- 两条路径每轮都实际完成前向、损失与指定顺序的清梯度/反向/更新，不能用注释或写死输出替代实验。
- 保存两条路径各自开始、结束的完整logits进行整体比较：形状为(4,2)，不能仅比较类别预测是否改变。
- 打印两条损失记录及各自开始/结束分数是否整体相同；实际报告候选顺序的表现，不设“必须提高准确率”的门槛。
- 给出运行前判断和实跑后解释；说明zero_grad允许提前到前向之前，但不能夹在backward和step之间清掉本轮所需梯度。

> 使用已有的torch.equal作整体精确比较即可；本题不引入新库。提交本课test.py、预测、真实输出和解释，不要求逐行注释。

### 我的练习解答
（在此粘贴你的代码和运行结果）

### 我的疑问
（记录没想通的点，会话里问我）

### 老师批改·第1轮（2026-11-15，批改对象：原test.py及教师修正版）

原作答区未贴代码/输出，最新提交以本目录test.py为准；原版已备份到 `teacher/原提交_20261115_190332/`。原版两循环共用已训练模型，缺完整logits比较、预测解释及本人输出。教师依本次推进授权直接修正，原版验收状态保留在批改文档，教师捕获不冒充本人输出。

**验收结论：教师协助修正后通过，本课作业关闭。** 两个干净副本和实际目录均退出码0，15项变量级断言通过且三次输出一致。演示回放与讲义自身代码均实跑通过。独立掌握不因此提升。

### 老师答疑

完整问题、准确修改位置、逐行注释参考答案、完整文字示范和原版/修正版真实输出见 [第十课教师批改与错误点参考答案](teacher/第十课_教师批改_2026-11-15.md)。test.py每处教师修改均标明原因；候选反向→清梯度→更新的顺序作为反例保留。课程venv缺torch，仅使用既有BaiQing3.10AI验证，未改环境。

### 补充答疑（2026-11-17）

梯度与优化器各做什么、清零为何能提前到前向之前、手写模型与TorchVision模型的关系，以及Normalize的具体含义，见[四问答疑与真实运行示例](补充答疑_梯度优化器模型与Normalize.md)。本次仅补充概念材料，不重新开启已关闭作业，不提升独立掌握状态。

## 【本节重点总结】

1. 模型前向产生 logits，损失函数比较 logits 与整数标签，自动求导计算梯度。
2. 前向必须先于损失与反向，反向先于更新；常规独立批次需在本次反向前清梯度。
3. SGD 与 Adam 都是用梯度更新参数的优化器，`backward` 负责算梯度，`step` 负责改参数。

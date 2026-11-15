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

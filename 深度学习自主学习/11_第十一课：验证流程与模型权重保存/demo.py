import torch  # 导入PyTorch张量和权重读写工具。
from torch import nn  # 导入模型基类、网络层和损失函数。
from torch.utils.data import TensorDataset, DataLoader  # 导入第九课的数据集与批次加载器。
torch.manual_seed(7)  # 固定随机种子，便于在同一环境复现初始化。
x_train = torch.tensor([[0.0, 0.2], [0.2, 0.0], [1.0, 1.2], [1.2, 1.0]], dtype=torch.float32)  # 训练特征，仅它参与反向更新。
y_train = torch.tensor([0, 0, 1, 1], dtype=torch.int64)  # 训练标签使用整数类别编号。
x_val = torch.tensor([[0.1, 0.1], [0.3, 0.0], [0.0, 0.3], [1.1, 1.1], [1.3, 1.0]], dtype=torch.float32)  # 验证特征不与训练样本重复。
y_val = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64)  # 验证标签只用于评价与选择权重。
x_test = torch.tensor([[0.05, 0.15], [1.15, 1.05]], dtype=torch.float32)  # 测试特征留到选好权重后再使用。
y_test = torch.tensor([0, 1], dtype=torch.int64)  # 测试标签不参与挑选最佳轮次。
val_dataset = TensorDataset(x_val, y_val)  # 沿用第九课：把对应位置的特征和标签配成样本。
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, drop_last=False)  # 保留末尾不满批次，避免漏算样本。
class SmallClassifier(nn.Module):  # 沿用第十课：定义可注册参数的模型。
    def __init__(self):  # 创建模型时初始化网络层。
        super().__init__()  # 沿用第十课：初始化模型基类。
        self.hidden = nn.Linear(2, 4)  # 沿用第十课：输入特征映射到中间特征。
        self.activation = nn.ReLU()  # 沿用第十课：添加非线性激活。
        self.output = nn.Linear(4, 2)  # 沿用第十课：输出每个类别的原始分数。
    def forward(self, inputs):  # 指定前向计算的连接顺序。
        hidden = self.hidden(inputs)  # 先完成第一个线性变换。
        activated = self.activation(hidden)  # 再把负值变成零。
        return self.output(activated)  # 返回logits（类别原始分数），无需提前转成概率。
model = SmallClassifier()  # 创建待训练模型。
loss_fn = nn.CrossEntropyLoss(reduction='mean')  # 函数⑤：得到每个批次的平均损失。
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)  # 沿用第十课：让优化器管理模型参数。
best_loss = float('inf')  # 辅助⑰：以正无穷初始化，使首轮的有限验证损失能入选。
best_epoch = 0  # 记录被选中权重对应的轮次。
checkpoint_path = 'demo_best.pt'  # 权重文件写在当前工作目录；重复运行会覆盖同名演示文件。
for epoch in range(8):  # 反复训练，每轮结束后验证同一份验证集。
    model.train()  # 函数①：每轮训练前明确恢复训练模式。
    train_logits = model(x_train)  # 训练前向需要建立计算图。
    train_loss = loss_fn(train_logits, y_train)  # 函数⑤：计算本轮更新前的训练损失。
    optimizer.zero_grad()  # 沿用第十课：反向前清除旧梯度。
    train_loss.backward()  # 沿用第十课：根据训练损失计算梯度。
    optimizer.step()  # 沿用第十课：利用梯度更新参数。
    model.eval()  # 函数②：开始验证前切到评估模式。
    loss_total = 0.0  # 累计所有验证样本的损失总量。
    correct_total = 0  # 累计预测正确的样本数。
    sample_total = 0  # 累计真正遍历的样本数。
    with torch.no_grad():  # 函数③：验证不建立用于反向传播的计算图。
        for features, targets in val_loader:  # 逐批验证，包括最后的不满批次。
            logits = model(features)  # 使用本轮更新后的同一个模型计算分数。
            loss = loss_fn(logits, targets)  # 函数⑤：当前批次的平均损失。
            batch_count = len(targets)  # 辅助⑰：取得本批实际样本数，不能一直使用预设batch_size。
            loss_total += loss.item() * batch_count  # 函数⑥：先还原本批损失总量，再纳入累计。
            predictions = logits.argmax(dim=1)  # 函数⑦：沿类别轴为每条样本选择最高分的编号。
            correct_total += (predictions == targets).sum().item()  # 函数⑧⑥：把逐样本真假比较汇成正确个数。
            sample_total += batch_count  # 累加本批样本数作为最终分母。
    val_loss = loss_total / sample_total  # 所有样本损失总量除以全部样本数，得到样本加权平均。
    val_accuracy = correct_total / sample_total  # 所有正确样本数除以全部样本数，得到accuracy（准确率）。
    if val_loss < best_loss:  # 本演示约定只按验证损失越小越好选权重；相等保留较早轮次。
        best_loss = val_loss  # 保存当前最佳的指标数值。
        best_epoch = epoch + 1  # 同步保存对应轮次。
        torch.save(model.state_dict(), checkpoint_path)  # 函数⑨⑩：立即写盘，固定本轮权重快照。
        with torch.no_grad():  # 函数③：留下一份最佳轮次的验证输出，之后核对重载是否恢复原行为。
            best_logits = model(x_val)  # 当前模型处于eval，得到独立输出张量，后续参数更新不会改写它。
    print('epoch', epoch + 1, 'train_loss', round(train_loss.item(), 6), 'val_loss', round(val_loss, 6), 'val_accuracy', round(val_accuracy, 6))  # 函数⑥及辅助⑰：打印实算指标。
reloaded = SmallClassifier()  # 创建新模型；它此刻尚未拥有最佳权重。
state = torch.load(checkpoint_path, weights_only=True, map_location='cpu')  # 函数⑪：按权重方式读入，并映射到CPU。
load_result = reloaded.load_state_dict(state, strict=True)  # 函数⑫：严格核对键名及形状，并复制到新模型。
reloaded.eval()  # 函数②：加载权重不会自动切换模式，推理前仍需明确设置。
with torch.inference_mode():  # 函数④：最终推理无需反向，并使用更严格的推理上下文。
    restored_logits = reloaded(x_val)  # 新模型输出整份验证集的原始分数。
    test_logits = reloaded(x_test)  # 最佳权重选定后，才进行最终测试。
    test_loss = loss_fn(test_logits, y_test).item()  # 函数⑤⑥：该测试集只用一个批次，直接取得平均损失。
    test_predictions = test_logits.argmax(dim=1)  # 函数⑦：从测试分数得到类别编号。
    test_accuracy = (test_predictions == y_test).sum().item() / len(y_test)  # 函数⑧⑥：计算最终测试准确率。
print('best_epoch', best_epoch, 'best_val_loss', round(best_loss, 6))  # 辅助⑰：展示按验证损失选中的轮次。
print('strict_load', load_result)  # 展示是否存在缺失键或多余键。
print('reloaded_logits_match', torch.allclose(restored_logits, best_logits, rtol=0.0, atol=0.0))  # 函数⑬：比较全部logits，不能只比较类别。
print('test_predictions', test_predictions.tolist())  # 函数⑭：将张量转成Python列表便于阅读。
print('test_loss', round(test_loss, 6), 'test_accuracy', round(test_accuracy, 6))  # 辅助⑰：测试结果是报告，不能倒过来挑权重。

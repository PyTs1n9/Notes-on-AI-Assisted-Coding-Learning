import torch
from torch import nn

torch.manual_seed(7)
x = torch.tensor([[0.0, 0.2], [0.2, 0.0], [1.0, 1.2], [1.2, 1.0]], dtype=torch.float32)
y = torch.tensor([0, 0, 1, 1])


class SmallClassifier(nn.Module):
	def __init__(self):
		super().__init__()
		self.hidden = nn.Linear(2, 4)
		self.activation = nn.ReLU()
		self.output = nn.Linear(4, 2)

	def forward(self, inputs):
		hidden = self.hidden(inputs)
		activate = self.activation(hidden)
		return self.output(activate)


# 老师新增：原来未写运行前判断 → 补上待检验的预测；此为教师参考，不算学生独立作答。
print("教师补充预测：候选顺序会清掉刚算出的梯度，模型分数应保持不变。")
model = SmallClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.5)
# 老师新增（以下两行）：原来缺训练前分数与损失记录 → 保留完整logits和列表，检验实际学习结果。
normal_start = model(x)  # 此输出是新计算出的张量，后续参数更新不会原地改变它的数值。
normal_losses = []  # 保存每轮更新前的真实损失。

for epoch in range(5):
	logits = model(x)
	loss = loss_fn(logits, y)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	# 老师新增：原来只有逐轮打印 → 同时记录实际loss，供两路径整体比较。
	normal_losses.append(loss.item())
	print("epoch:", epoch, "loss", round(loss.item(), 4))

# 老师新增：原来未保存训练后完整分数 → 重新前向，避免把最后一次更新前的logits当成结束状态。
normal_end = model(x)
print()

# 老师修改（以下三行）：原来第二循环沿用已训练model和optimizer → 同种子重建独立对象；使对照只改变清梯度顺序。
torch.manual_seed(7)  # 恢复与第一条路径相同的初始化随机状态。
candidate_model = SmallClassifier()  # 独立模型，不继承第一条路径训练结果。
candidate_optimizer = torch.optim.SGD(params=candidate_model.parameters(), lr=0.5)  # 优化器必须管理候选模型参数。
# 老师新增（以下两行）：原来缺候选起点和记录 → 保存完整分数与每轮loss。
candidate_start = candidate_model(x)
candidate_losses = []

for epoch in range(5):
	# 老师修改（以下五行）：原来仍引用第一条路径对象 → 改用候选对象；保留题目要求的错误顺序作反例。
	logits = candidate_model(x)  # 只改变顺序，不改变模型、数据和学习率。
	loss = loss_fn(logits, y)  # 仍使用同一个无可训练参数的交叉熵损失函数。
	loss.backward()  # 计算新梯度。
	candidate_optimizer.zero_grad()  # 故意清掉刚得到的梯度，观察后续更新后果。
	candidate_optimizer.step()  # 默认清梯度后grad为None，SGD跳过这些参数。
	# 老师新增：原来缺候选完整损失记录 → 保存真实值，不能写死预期结论。
	candidate_losses.append(loss.item())
	print("epoch:", epoch, "loss", round(loss.item(), 4))

# 老师新增（以下整个证据块）：原来缺完整分数整体比较和事后解释 → 用torch.equal实际比较，再提供教师参考解读。
candidate_end = candidate_model(x)  # 完成最后一轮后重新计算分数。
print("正常路径损失记录", normal_losses)  # 显示两条独立路径的更新前损失。
print("候选路径损失记录", candidate_losses)
print("正常起点分数", normal_start.tolist())  # 完整分数而非类别预测，避免argmax不变掩盖参数变化。
print("正常终点分数", normal_end.tolist())
print("候选起点分数", candidate_start.tolist())
print("候选终点分数", candidate_end.tolist())
print("起点是否相同", torch.equal(normal_start, candidate_start))  # 确认公平对照。
print("正常起止形状", normal_start.shape, normal_end.shape)
print("候选起止形状", candidate_start.shape, candidate_end.shape)
print("正常起止分数整体相同", torch.equal(normal_start, normal_end))  # 由实际数值给出布尔结果。
print("候选起止分数整体相同", torch.equal(candidate_start, candidate_end))
print("教师补充解释：backward计算梯度，step使用梯度更新参数；中间清梯度会丢掉本轮更新依据。")
print("教师补充解释：zero_grad可提前到前向之前，只要本次backward前清旧梯度，step前保留新梯度。")

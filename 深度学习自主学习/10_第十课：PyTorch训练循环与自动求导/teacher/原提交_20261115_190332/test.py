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


model = SmallClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.5)

for epoch in range(5):
	logits = model(x)
	loss = loss_fn(logits, y)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	print("epoch:", epoch, "loss", round(loss.item(), 4))

print()

for epoch in range(5):
	logits = model(x)
	loss = loss_fn(logits, y)
	loss.backward()
	optimizer.zero_grad()
	optimizer.step()
	print("epoch:", epoch, "loss", round(loss.item(), 4))

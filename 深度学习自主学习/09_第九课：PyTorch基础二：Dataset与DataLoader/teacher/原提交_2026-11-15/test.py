import torch
from torch.utils.data import Dataset, DataLoader

samples = [
	[[[1.0, 2.0], [3.0, 4.0]]],
	[[[5.0, 6.0], [7.0, 8.0]]],
	[[[9.0, 10.0], [11.0, 12.0]]],
	[[[13.0, 14.0], [15.0, 16.0]]],
	[[[17.0, 18.0], [19.0, 20.0]]],
	[[[21.0, 22.0], [23.0, 24.0]]],
]
labels = [1, 0, 1, 0, 1, 0]

images = torch.tensor(samples, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.int64)

train_images = images[:5]
train_labels = labels[:5]
val_images = images[5:]
val_labels = labels[5:]
train_mean = train_images.mean()
train_std = train_images.std()


class ImageDataset(Dataset):
	def __init__(self, images, labels, mean, std, normalize=True):
		self.images = images
		self.labels = labels
		self.mean = mean
		self.std = std
		self.normalize = normalize

	def __getitem__(self, index):
		image = self.images[index]
		label = self.labels[index]
		if self.normalize:
			image = (image - self.mean) / self.std
		return image, label

	def __len__(self):
		return self.images.shape[0]


train_dataset = ImageDataset(images, labels, train_mean, train_std)
val_dataset = ImageDataset(images, labels, train_mean, train_std)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)

normalized_image, normalized_label = train_dataset[0]
print('单样本：', normalized_label)
print('归一化首样本首像素：', normalized_image[0, 0, 0])
print('Dataset长度：', len(train_dataset), len(val_dataset))

for batch_images, batch_labels in train_loader:
	print('训练批次：', batch_images.shape, batch_images.dtype, batch_labels.tolist())
	break

for batch_images, batch_labels in val_loader:
	print('验证批次：', batch_images.shape, batch_labels.tolist())
	break

print('Loader批次数：', len(train_loader), len(val_loader))

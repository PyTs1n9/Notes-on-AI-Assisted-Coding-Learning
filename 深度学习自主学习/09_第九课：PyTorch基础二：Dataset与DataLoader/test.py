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

# 老师修改（以下切分与统计块）：原连续切片不符合指定索引 → 改为同步索引图像和标签，保证集合互斥且顺序正确。
train_indices = [0, 2, 3, 5]  # 保存训练样本的原始身份。
val_indices = [4, 1]  # 保留题目指定的验证顺序。
train_images = images[train_indices]  # 只取训练图像。
train_labels = labels[train_indices]  # 标签使用同一组索引。
val_images = images[val_indices]  # 只取验证图像。
val_labels = labels[val_indices]  # 同步验证标签。
train_mean = train_images.mean()  # 仅训练样本参与统计。
train_std = train_images.std(unbiased=False)  # 老师修改：原默认样本标准差 → 总体口径，与题目一致。


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


# 老师修改（数据集与加载器块）：原两侧均传全量数据且验证也打乱 → 分别传切分结果，按题目设置批次与尾批次策略。
train_dataset = ImageDataset(train_images, train_labels, train_mean, train_std)  # 只提供训练样本。
val_dataset = ImageDataset(val_images, val_labels, train_mean, train_std)  # 验证复用训练统计量。
train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, drop_last=True, num_workers=0)  # 丢弃打乱后的不完整尾批次。
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, drop_last=False, num_workers=0)  # 保持验证索引顺序。

# 老师修改（单样本证据块）：原只输出训练标签和像素 → 两侧均输出形状、类型、设备和归一化像素，补齐证据。
print('训练统计量：', train_mean, train_std)  # 所有统计量只取自训练集。
print('Dataset长度：', len(train_dataset), len(val_dataset))  # 核对真实数据集长度。
for name, dataset in [('训练', train_dataset), ('验证', val_dataset)]:  # 分别观察两个集合。
    image, label = dataset[0]  # 读取单样本，不能预先加入B轴。
    print(name, '单样本：', image.shape, image.dtype, image.device, label, label.dtype, label.device)  # 显示图像和标签契约。
    print(name, '归一化首像素：', image[0, 0, 0])  # 观察真实归一化结果。

# 老师新增（种子调查块）：原仅取首批且未追踪样本身份 → 完整遍历两次并以全图比较追踪原始索引，避免重复标签混淆身份。
print('教师补充预测：该断言不成立；丢弃的是本次打乱后的尾批次，不固定对应原顺序最后一个样本。')  # 原提交未作预测，此处是教师参考。
for seed in [7, 21]:  # 使用题目指定的两个种子。
    kept = []  # 每次重新记录保留者。
    torch.manual_seed(seed)  # 紧接着遍历加载器，其间不执行其他随机操作。
    for batch_images, batch_labels in train_loader:  # 完整遍历，不提前break。
        print('训练批次：', seed, batch_images.shape, batch_images.dtype, batch_images.device, batch_labels.shape, batch_labels.dtype, batch_labels.device, batch_labels.tolist())  # 补齐批次证据。
        for image in batch_images:  # 逐个追踪本批样本。
            for original_index in train_indices:  # 只在训练身份中查找。
                expected_image = (images[original_index] - train_mean) / train_std  # 使用同一规则构造候选图像。
                if torch.equal(image, expected_image):  # 老师新增：全图比较辨认身份，torch.equal返回整体是否完全一致。
                    kept.append(original_index)  # 记录实际打乱后的身份顺序。
    dropped = []  # 记录未出现的训练身份。
    for original_index in train_indices:  # 逐一检查训练样本。
        if original_index not in kept:  # 不在保留者中即为被丢弃者。
            dropped.append(original_index)  # 保存其原始索引。
    print('样本身份：', seed, '保留', kept, '丢弃', dropped)  # 不能用重复的0/1标签代替身份。

# 老师新增：原验证仅打标签且shuffle错误 → 完整读取验证批次，核对顺序与全部类型。
for batch_images, batch_labels in val_loader:  # 验证侧不打乱、不丢弃。
    print('验证批次：', batch_images.shape, batch_images.dtype, batch_images.device, batch_labels.shape, batch_labels.dtype, batch_labels.device, batch_labels.tolist())  # 验证标签对应原索引[4,1]。
print('Loader批次数：', len(train_loader), len(val_loader))  # 显示批次层数量。
# 老师新增（原解释未提交）：Dataset按索引提供一张CHW图像及一个标签；DataLoader按采样顺序把多条样本叠成BCHW及一批标签。
# 老师新增（原预测复核未提交）：保留/丢弃者见上方真实结果；shuffle改变采样顺序，drop_last丢的是采样后的尾批次；验证标签保持[4,1]对应的顺序。
# 老师新增：torch.equal(input, other)返回Python布尔值；要求形状与元素全等，适合同一计算产生的精确身份追踪；浮点近似比较应另用容差方法。

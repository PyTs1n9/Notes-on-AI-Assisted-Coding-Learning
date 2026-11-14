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

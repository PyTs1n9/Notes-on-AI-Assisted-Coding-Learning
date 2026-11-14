import torch  # 导入本课使用的张量计算库。

data = [
	[[11, 21, 31], [12, 22, 32]],  # 第0行像素行，2个像素点
	[[13, 23, 33], [14, 24, 34]]  # 第1行像素行，2个像素点
]  # 行、列、RGB的嵌套顺序。
hwc = torch.tensor(data, dtype=torch.uint8, device='cpu')  # 函数①：明确按原始像素类型创建HWC张量。
floating = hwc.to(dtype=torch.float32)  # 函数⑩：只改数值类型，不自动缩放像素。
regrouped = torch.reshape(floating, (3, 2, 2))  # 函数④：把原有数字按目标形状重新分组。
chw = torch.permute(floating, (2, 0, 1))  # 函数⑤：新轴依次来自旧的通道、行、列轴。
bchw = torch.unsqueeze(chw, 0)  # 函数⑥：在最前面补上单张样本的批次轴。
print('创建/改类型：', hwc.dtype, floating.dtype, torch.equal(hwc, floating))  # 函数②⑦：另查dtype，比较数字是否保持。
print('重组/换轴shape：', regrouped.shape, chw.shape)  # 函数②：先看两种结果的形状。
print('重组/换轴全量相等：', torch.equal(regrouped, chw))  # 函数⑦：相同形状不能代替逐元素核对。
print('原像素/重组后/换轴后：', floating[0, 1, :], regrouped[:, 0, 1], chw[:, 0, 1])  # 函数②：追踪同一行列坐标的颜色分量。
print('批次shape/首张相等：', bchw.shape, torch.equal(bchw[0], chw))  # 函数②⑦：新增轴后原图数据仍对应。

device = torch.device('cpu')  # 函数⑨：默认选CPU，保证无CUDA也能学完。
if torch.cuda.is_available():  # 函数⑧：只有检测可用时才选择CUDA。
	device = torch.device('cuda:0')  # 函数⑨：描述实际存在的CUDA目标。
on_device = bchw.to(device=device)  # 函数⑩：接住设备转换结果，不改变原变量。
back_on_cpu = on_device.to(device='cpu')  # 函数⑩：回CPU后用一致设备核对数值。
print('原设备/目标设备/往返相等：', bchw.device, on_device.device, torch.equal(back_on_cpu, bchw))  # 函数②⑦：确认搬运后数字保持。

import torch  # 导入本题使用的PyTorch。

pixels = [[[3, 22, 60], [9, 45, 80], [18, 65, 95]], [[25, 15, 120], [40, 35, 140], [55, 75, 160]]]  # 保留题目给定的HWC、RGB像素。
offsets = [8, 30, 70]  # 保留题目给定的RGB通道偏移。

hwc_pixels = torch.tensor(pixels, dtype=torch.uint8, device='cpu')  # 函数①：按原始像素类型创建HWC张量。
hwc_offsets = torch.tensor(offsets, dtype=torch.uint8, device='cpu')  # 函数①：按原始整数类型创建偏移张量。
print('输入shape/dtype：', hwc_pixels.shape, hwc_pixels.dtype, hwc_offsets.shape, hwc_offsets.dtype)  # 函数②：核对两个原始输入。

# 老师修改：原来只把像素转成float32，偏移仍是uint8；现把两个减法操作数一起转成float32，避免负数下溢。
pixels_float = hwc_pixels.to(dtype=torch.float32)  # 函数⑩：先把像素转为能保存负数的类型。
offsets_float = hwc_offsets.to(dtype=torch.float32)  # 函数⑩：先把偏移转为相同的计算类型。
# 老师修改：原提交预测“先相减再改浮点”是错误方向；改为先让两个操作数都能表示负数，再相减。

# 老师新增：保留“先用uint8计算、再转float”的对照，只用于证明错误结果无法靠事后转换补救。
integer_result = hwc_pixels - hwc_offsets  # 两个操作数都是uint8，负数会在无符号整数范围内回绕。
late_float = integer_result.to(dtype=torch.float32)  # 函数⑩：事后转float只转换已经算出的错误数字。
correct_hwc = pixels_float - offsets_float  # 两个float32操作数相减，正确保留负值。
print('先uint8计算再转float的首像素：', late_float[0, 0, :])  # 函数②：显示错误路径的首个RGB像素。
print('先转float再计算的首像素：', correct_hwc[0, 0, :])  # 函数②：显示正确路径保留的负值。

pixels_chw = torch.permute(pixels_float, (2, 0, 1))  # 函数⑤：把HWC换成通道、行、列。

# 老师新增：保留一维偏移的错误对齐结果；它能运行且shape相同，但广播会从最右侧对齐到宽度轴。
wrong_chw = pixels_chw - offsets_float  # 错误示范：长度3的偏移实际对齐到CHW的最后一根宽度轴。

# 老师新增：原来没有把RGB偏移明确对齐到CHW通道轴；补成(通道,1,1)后，偏移才按颜色通道作用。
channel_offsets = torch.reshape(offsets_float, (3, 1, 1))  # 函数④：为RGB通道补两根长度为1的轴。
correct_chw = pixels_chw - channel_offsets  # 按CHW的第0根轴逐通道相减。
correct_from_chw = torch.permute(correct_chw, (1, 2, 0))  # 函数⑤：换回HWC以便和正确HWC结果比较。
print('正确HWC/CHW全像素一致：', torch.equal(correct_hwc, correct_from_chw))  # 函数⑦：比较全部像素，不只比较shape。
print('同一像素正确RGB/错误对齐结果：', correct_chw[:, 0, 1], wrong_chw[:, 0, 1])  # 函数②：检查一个行列位置的颜色对应。
print('错误对齐后shape仍相同/全量相等：', wrong_chw.shape, torch.equal(wrong_chw, correct_chw))  # 函数⑦：证明能运行且shape相同仍可能算错。

pixels_b_chw = torch.unsqueeze(correct_chw, 0)  # 函数⑥：给正确CHW结果补上单张图像的批次轴。

# 老师修改：原来device只在CUDA分支里创建；先设CPU默认值，才能在无CUDA机器上继续运行。
device = torch.device('cpu')  # 函数⑨：设置所有环境都可用的默认设备。
if torch.cuda.is_available():  # 函数⑧：只有检测到CUDA可用时才切换设备。
	device = torch.device('cuda:0')  # 函数⑨：描述当前可用的第一块CUDA设备。
on_device = pixels_b_chw.to(device=device)  # 函数⑩：接住设备转换结果，不改变原变量。
back_on_cpu = on_device.to(device='cpu')  # 函数⑩：把结果搬回CPU后再比较。
print('结果shape/dtype：', pixels_b_chw.shape, pixels_b_chw.dtype)  # 函数②：核对最终批次结果。
print('设备/往返一致：', pixels_b_chw.device, on_device.device, torch.equal(back_on_cpu, pixels_b_chw))  # 函数⑦：核对设备和全部数值保持。

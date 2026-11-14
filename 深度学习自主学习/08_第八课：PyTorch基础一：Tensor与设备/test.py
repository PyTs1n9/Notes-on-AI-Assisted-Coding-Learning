import torch

pixels = [[[3, 22, 60], [9, 45, 80], [18, 65, 95]], [[25, 15, 120], [40, 35, 140], [55, 75, 160]]]
offsets = [8, 30, 70]

hwc_pixels = torch.tensor(pixels, dtype=torch.uint8, device='cpu')
floating = hwc_pixels.to(dtype=torch.float32)  # 函数⑩：只改数值类型，不自动缩放像素。
hwc_offsets = torch.tensor(offsets, dtype=torch.uint8, device='cpu')
# 预测：不认同，应该先相减再改浮点，偏移后位置就不对了，相减是对应的位置。
print(hwc_pixels.shape, hwc_pixels.dtype)
print(hwc_offsets.shape, hwc_offsets.dtype)

pixels_j = hwc_pixels - hwc_offsets
pixels_chw = torch.permute(floating, (2, 0, 1))
pixels_b_chw = torch.unsqueeze(pixels_chw, 0)
print(pixels_b_chw.shape, pixels_b_chw.dtype)
if torch.cuda.is_available():
	device = torch.device('cuda')
on_device = pixels_b_chw.to(device=device)  # 函数⑩：接住设备转换结果，不改变原变量。
back_on_cpu = on_device.to(device='cpu')  # 函数⑩：回CPU后用一致设备核对数值。

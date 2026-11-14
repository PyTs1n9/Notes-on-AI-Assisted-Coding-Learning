import json  # 教师私有核验：保存实跑计算与断言，不随正式课程发布。
import sys  # 读取是否只验证CPU的参数。
import torch  # 本课张量库；不下载数据或权重。
pixels = [[[3, 22, 60], [9, 45, 80], [18, 65, 95]], [[25, 15, 120], [40, 35, 140], [55, 75, 160]]]  # 与最终题面相同的材料。
offsets = [8, 30, 70]  # RGB对应偏移材料。
raw = torch.tensor(pixels, dtype=torch.uint8, device='cpu')  # 函数①：保留原始整数输入。
offset_byte = torch.tensor(offsets, dtype=torch.uint8, device='cpu')  # 函数①：隔离整数计算的影响。
late_integer = raw - offset_byte  # 函数③：先在不带负号的整数中运算。
late_float = late_integer.to(dtype=torch.float32)  # 函数⑩：运算之后改类型，观察能否补救。
floating = raw.to(dtype=torch.float32)  # 函数⑩：提前转换，使负数可表示。
offset_float = offset_byte.to(dtype=torch.float32)  # 函数⑩：比较时统一元素类型。
early_hwc = floating - offset_float  # 函数③：HWC的最后轴正好是通道。
correct_chw = torch.permute(early_hwc, (2, 0, 1))  # 函数⑤：第一条安全路径作为坐标基准。
original_chw = torch.permute(floating, (2, 0, 1))  # 函数⑤：第二条路径先换轴，隔离广播对齐问题。
wrong_chw = original_chw - offset_float  # 函数③：可运行但向量对齐最后的宽轴。
channel_offsets = torch.reshape(offset_float, (3, 1, 1))  # 函数④：把偏移的通道轴显式放到对应位置。
alternative_chw = original_chw - channel_offsets  # 函数③：在CHW上使用明确的逐通道偏移。
batch = torch.unsqueeze(alternative_chw, 0)  # 函数⑥：补批次轴，并保持颜色对应。
device = torch.device('cpu')  # 函数⑨：默认CPU即可完成。
if '--auto' in sys.argv and torch.cuda.is_available():  # 函数⑧：只有明确要求且实际可用才搬到CUDA。
    device = torch.device('cuda:0')  # 函数⑨：选择可用CUDA设备。
result = batch.to(device=device)  # 函数⑩：接住返回值。
back = result.to(device='cpu')  # 函数⑩：统一设备后核验。
checks = {}  # 后续均为教师核验，不要求学生实现这些工具。
checks['input_preserved'] = torch.equal(raw, torch.tensor(pixels, dtype=torch.uint8, device='cpu'))
checks['input_uint8'] = raw.dtype == torch.uint8
checks['late_cast_does_not_repair'] = not torch.equal(late_float, early_hwc)
checks['negative_values_retained'] = bool((early_hwc < 0).any().item())
checks['wrong_alignment_runs_with_same_shape'] = wrong_chw.shape == correct_chw.shape
checks['wrong_alignment_not_equivalent'] = not torch.equal(wrong_chw, correct_chw)
checks['two_correct_orders_agree'] = torch.equal(alternative_chw, correct_chw)
checks['output_bchw_shape'] = list(result.shape) == [1, 3, 2, 3]
checks['output_float32'] = result.dtype == torch.float32
checks['batch_data_preserved'] = torch.equal(back[0], correct_chw)
checks['move_device_correct'] = result.device == device
checks['roundtrip_values_preserved'] = torch.equal(back, batch)
checks['plain_vector_was_aligned_to_width'] = torch.equal(wrong_chw[:, 0, 0], original_chw[:, 0, 0] - offset_float[0])
all_coordinates = True
for h in range(raw.shape[0]):
    for w in range(raw.shape[1]):
        if not torch.equal(back[0, :, h, w], floating[h, w, :] - offset_float):
            all_coordinates = False
checks['every_pixel_coordinate_and_channel_matches'] = all_coordinates
assert all(checks.values()), checks
payload = {'mode': 'auto' if '--auto' in sys.argv else 'cpu', 'torch_version': torch.__version__,
           'cuda_available': torch.cuda.is_available(), 'input_shape': list(raw.shape),
           'output_shape': list(result.shape), 'output_dtype': str(result.dtype), 'output_device': str(result.device),
           'input_pixel': raw[0, 0, :].tolist(), 'late_cast_pixel': late_float[0, 0, :].tolist(),
           'early_cast_pixel': early_hwc[0, 0, :].tolist(), 'wrong_alignment_pixel': wrong_chw[:, 0, 0].tolist(),
           'correct_pixel': back[0, :, 0, 0].tolist(), 'full_output_cpu': back.tolist(),
           'checks': checks, 'assertions_passed': len(checks)}
print(json.dumps(payload, ensure_ascii=False, indent=2))

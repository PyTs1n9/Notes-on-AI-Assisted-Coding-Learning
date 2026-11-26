# 用低通保留慢变化、压制高频扰动：相同长度的输入和输出做频率成分对比，验证滤波确实保留低频。
# 构造已知由慢信号与高频扰动叠加的记录。
import numpy as np
# 设计双二阶节滤波器并离线前后滤波。
from scipy.signal import butter, sosfiltfilt

# 每秒 1000 个采样点。
sr = 1000
# 相邻采样间隔为 1/sr 秒。
t = np.arange(sr) / sr
# 需要保留的慢变化成分。signal：一个每秒振动 35 次的波
signal = np.sin(2 * np.pi * 35 * t)
# 模拟高频扰动。noise ：一个每秒振动 230 次的波，高度是前者的 0.8 倍
noise = 0.8 * np.sin(2 * np.pi * 230 * t)
# 对混合信号滤波，不去访问纯净信号做预测。mixed ：把这两个波加起来
mixed = signal + noise
# 截止频率 80 Hz；明示单位。butter巴特沃斯滤波器
sos = butter(
	4,  # 阶数越高，截止频率附近的过渡通常越陡，高频衰减越快
	80,  # 截止频率
	btype='lowpass',  # 低通滤波器
	fs=sr,  # 指定采样率为 1000 Hz，因此这里的 80 按 Hz 理解
	output='sos'  # 把滤波器参数保存为 SOS 格式
)
# 适用于完整离线信号。
filtered = sosfiltfilt(sos, mixed)
# 比较滤波前的正频率幅值。
before = np.abs(np.fft.rfft(mixed))
# 同长度、同频率坐标下比较滤波后的幅值。
after = np.abs(np.fft.rfft(filtered))
# 根据采样间隔建立 Hz 轴。
freq = np.fft.rfftfreq(len(mixed), d=1 / sr)
# 比较目标信号和扰动，不能只看图是否变平。
for hz in [35, 230]:
	# 找目标 Hz 的频率下标。
	i = int(np.argmin(np.abs(freq - hz)))
	# 使用相同尺度作相对比较。
	print('频率、滤波前后幅值', hz, round(float(before[i]), 2), round(float(after[i]), 2))

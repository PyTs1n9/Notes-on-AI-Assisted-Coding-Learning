# find_peaks 是 SciPy 里的一个函数，用来找数据里的“山顶”：比左右两个邻居都高的点。
from scipy.signal import find_peaks

data = [0, 3, 0, 1, 0]

peaks, properties = find_peaks(data)

# [1 3]：峰顶的位置（下标从 0 开始）
print(peaks)
# [3, 1]：峰顶的数值
print([data[i] for i in peaks])

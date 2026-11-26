import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

data = np.array([0, 1, 0, 6, 0, 4, 0, 0, 5, 0, 1, 0])

peaks, properties = find_peaks(
    data,
    height=3,    # 峰的数值至少为 3
    distance=3   # 保留的峰，下标之间至少相差 3
)

print("峰的位置：", peaks)                       # [3 8]
print("峰的高度：", properties["peak_heights"])  # [6. 5.]

plt.plot(data, "o-", label="Signal")
plt.plot(peaks, data[peaks], "rx", markersize=12, label="Peaks")
plt.axhline(3, color="gray", linestyle="--", label="Min height")
plt.legend()
plt.show()
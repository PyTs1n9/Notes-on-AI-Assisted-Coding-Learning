# SciPy 科学计算：峰值、滤波与稳健统计

[返回示例总目录](../README.md)。每段独立运行，下面是 `BaiQing3.10AI` 的真实输出。

## 函数卡

- **作用**：SciPy 是库名，不是函数；这里练信号找峰、数字滤波和对异常点不敏感的统计。
- **语法**：`scipy.signal.find_peaks(x, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)`；`scipy.signal.butter(N, Wn, btype="low", analog=False, output="ba", fs=None)`；`scipy.signal.sosfiltfilt(sos, x, axis=-1, padtype="odd", padlen=None)`；`scipy.stats.median_abs_deviation(x, axis=0, center=np.median, scale=1.0, nan_policy="propagate")`。
- **关键参数**：`distance` 的单位是样本下标间距；`prominence` 衡量局部峰突出程度；`fs` 指定采样率后滤波截止频率用 Hz；MAD（中位数绝对偏差）默认返回未缩放的稳健散布量。
- **返回值**：`find_peaks` 返回峰下标与属性字典；`butter` 在 `output="sos"` 时返回可交给滤波器的系数；`sosfiltfilt` 返回滤波后的数组。
- **常见坑**：局部最大值不一定是业务有效峰；滤波截止频率应小于采样率的一半；前后向滤波适合离线，不能直接当实时算法。MAD 若为 0，不能直接拿来做除数。
- **适用场景**：离线设备监测、传感器信号过滤、鲁棒的异常标记。

## 示例 1：过滤设备曲线上的微小假峰

只报告有足够突出程度、且间距足够的周期冲击。

运行 [示例01.py](示例01.py)。

```text
全部局部峰 [1, 3, 5, 7, 9, 11, 13]
保留的冲击位置 [3, 9, 13]
保留峰的突出程度 [1.4, 1.6, 1.3]
```

## 示例 2：用低通保留慢变化、压制高频扰动

相同长度的输入和输出做频率成分对比，验证滤波确实保留低频。

运行 [示例02.py](示例02.py)。

```text
频率、滤波前后幅值 35 500.0 499.64
频率、滤波前后幅值 230 400.0 0.71
```

## 示例 3：用 MAD 找异常处理耗时

极大故障点会拉高均值与标准差；这里以中位数和 MAD 为口径报告异常。

运行 [示例03.py](示例03.py)。

```text
中位数、MAD 21.0 1.0
长耗时告警 [33, 170]
```

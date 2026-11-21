# 华为杯 · 14 项专项示例

每项 3 个独立 `.py`，共 42 个。沿用 PyCharm 的 `BaiQing3.10AI`，直接运行任意脚本，不必按顺序执行，也无需下载数据。

代码使用的音频和内嵌数据都是教学数据；图示例保存在对应专题的 `outputs/`，不要预先创建。每个专题的说明含函数卡和实际运行输出。

| 目录 | 重点 | 3 个案例 |
| --- | --- | --- |
| [0_jieba](0_jieba/README.md) | 结巴中文分词 | 精确分词和搜索索引；反馈分词后滤停用词；加入设备专有词 |
| [1_TfidfVectorizer](1_TfidfVectorizer/README.md) | TF-IDF 文本向量化 | 看出常用模板词权重；新文本遇到单字和未知词；加上词组区分相反短语 |
| [2_LogisticRegression](2_LogisticRegression/README.md) | 逻辑回归分类器 | 订单是否需人工复核；比较两个预先给定的预警阈值；多分类概率按类名取列 |
| [3_str.contains](3_str.contains/README.md) | Pandas 字符串查找 | 按字面搜索版本片段；筛选有格式的报警编码；包含故障但排除演练工单 |
| [4_str.translate](4_str.translate/README.md) | Python 与 Pandas 字符映射 | 只删除编号中的分隔字符；分隔词时替换为空格而不删除；用字典统一全角文本 |
| [5_PCA](5_PCA/README.md) | PCA 主成分分析 | 由解释方差比例决定保留几列；新记录复用历史坐标轴；同一数据先不缩放再缩放 |
| [6_Seaborn](6_Seaborn/README.md) | Seaborn 按字段表达数据 | 簇编号着色散点图；按行归一化的差错热图；用箱线图检查同中位数不同波动 |
| [7_histogram](7_histogram/README.md) | NumPy 数值直方图 | RGB 三通道各自计数；按质检区间统计厚度；不等宽直方图的密度和箱内比例 |
| [8_Sobel](8_Sobel/README.md) | OpenCV 有方向 Sobel 梯度 | 定位竖条的左右边；保留暗变亮和亮变暗；平滑后再查找带噪图像的边 |
| [9_magnitude](9_magnitude/README.md) | OpenCV 向量幅值 | 找出跑带上的二维移动异常；找出只看横向会漏掉的边；用各区域占比比较表面纹理 |
| [10_SciPy](10_SciPy/README.md) | SciPy 科学计算：峰值、滤波与稳健统计 | 过滤设备曲线上的微小假峰；用低通保留慢变化、压制高频扰动；用 MAD 找异常处理耗时 |
| [11_wavfile.read](11_wavfile.read/README.md) | 读 WAV 与识别采样格式 | 读取单声道 PCM16 并汇报时长；读取立体声，观察混成单声道后的抵消；把 uint8 采样还原成以零为中心的波形 |
| [12_fft](12_fft/README.md) | NumPy 快速傅里叶变换 | 从双音信号取出两个主频与幅度；短音频补零后比较频率格点；对称删掉一对干扰频点再还原 |
| [13_fftfreq](13_fftfreq/README.md) | NumPy FFT 频率坐标 | 从 5 毫秒记录间隔建立频率轴；不同采样率下，同一音对应不同下标；偶数点 FFT 的 Nyquist 频点不在正频掩码内 |

[返回代码填空练习](../卷子/代码填空练习/README.md)。

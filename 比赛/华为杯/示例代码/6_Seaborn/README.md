# Seaborn 按字段表达数据

[返回示例总目录](../README.md)。每段独立运行，下面是 `BaiQing3.10AI` 的真实输出。

## 函数卡

- **作用**：把数据列映射到横轴、纵轴、颜色等视觉属性。
- **语法**：`sns.scatterplot(data=None, *, x=None, y=None, hue=None, size=None, style=None, palette=None, hue_order=None, hue_norm=None, sizes=None, size_order=None, size_norm=None, markers=True, style_order=None, legend="auto", ax=None, **kwargs)`；`sns.heatmap(data, *, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt=".2g", annot_kws=None, linewidths=0, linecolor="white", cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels="auto", yticklabels="auto", mask=None, ax=None, **kwargs)`；`sns.boxplot` 常用调用为 `sns.boxplot(data=frame, x="线路", y="耗时", ax=ax)`。
- **关键参数**：`hue` 指定颜色所代表的分组列；热图 `annot=True` 显示数值，`fmt` 选择数字格式；`ax` 指定画到哪个子图。
- **返回值**：这些坐标轴级函数返回对应的 Matplotlib Axes（坐标区），可加标题，也能从原 Figure 保存。
- **常见坑**：整型标签用离散调色板，颜色按组理解。热图传入的二维表要先做好行列含义；盒图的须端不是简单的全组最值。
- **适用场景**：聚类散点、汇总矩阵热图、分组分布对比。

## 示例 1：簇编号着色散点图

把同一批二维坐标按簇着色，避免误把坐标本身用于着色。

运行 [示例01.py](示例01.py)。

输出文件：`outputs/clusters.png`。

```text
每组样本数 {'A': 3, 'B': 3}
```

## 示例 2：按行归一化的差错热图

组规模不同，比较每组类别占比而非原人数。

运行 [示例02.py](示例02.py)。

输出文件：`outputs/line_rates.png`。

```text
各线重试占比 {'Line A': 0.1, 'Line B': 0.4}
```

## 示例 3：用箱线图检查同中位数不同波动

两条线路的典型送达时间可以一样，但其中一条明显更不稳定。

运行 [示例03.py](示例03.py)。

输出文件：`outputs/delivery_spread.png`。

```text
线路、中位数、IQR Stable 20.0 2.0
线路、中位数、IQR Variable 20.0 17.5
```

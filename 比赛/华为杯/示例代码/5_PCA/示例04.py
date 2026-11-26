from sklearn.decomposition import PCA

# 4 条数据，每条有 2 个特征
# 第二列始终是第一列的两倍，信息重复
X = [
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8],
]

pca = PCA(n_components=1)  # 压缩到 1 维
result = pca.fit_transform(X)

print("压缩后的数据：")
print(result.round(2))

print("保留的信息比例：")
# explained_variance_ratio_ 解释变量比率_
print(pca.explained_variance_ratio_)
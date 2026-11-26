from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 两列特征的数值范围差异很大
X = [
    [1, 100],
    [2, 200],
    [3, 300],
    [4, 400],
]

# 1. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("标准化后：")
print(X_scaled.round(2))
# [[-1.34 -1.34]
#  [-0.45 -0.45]
#  [ 0.45  0.45]
#  [ 1.34  1.34]]

# 2. PCA：从 2 维压缩成 1 维
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

print("降维后：")
print(X_pca.round(2))

print("保留的方差比例：")
print(pca.explained_variance_ratio_)  # [1.]，即 100%
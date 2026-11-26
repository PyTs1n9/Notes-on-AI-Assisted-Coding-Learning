from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

X = [
    [1, 100],
    [2, 200],
    [3, 300],
    [4, 400],
]

# 把两个步骤串起来
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=1)),
])

# 从已有数据中学习标准化规则和降维方向，然后转换
X_pca = pipeline.fit_transform(X)
print("已有数据降维后：")
print(X_pca.round(2))

# 新数据沿用已经学到的规则
X_new = [
    [5, 500],
    [6, 600],
]

X_new_pca = pipeline.transform(X_new)
print("新数据降维后：")
print(X_new_pca.round(2))
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# 150 条数据，每条有 4 个特征，加载内置的鸢尾花示例数据集
X = load_iris().data

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
])

X_pca = pipeline.fit_transform(X)

# 取出流水线中训练好的 PCA
pca = pipeline.named_steps["pca"]

print("原始形状：", X.shape)
print("降维形状：", X_pca.shape)
print("自动选择的维数：", pca.n_components_)
print("各主成分的方差比例：", pca.explained_variance_ratio_.round(4))
print(f"累计保留方差：{pca.explained_variance_ratio_.sum():.2%}")
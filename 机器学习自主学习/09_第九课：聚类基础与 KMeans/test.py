import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('customers.csv', encoding='utf-8-sig')
print(df.shape)
print(df.dtypes)
print(df.isnull().sum().sum())

scaler = StandardScaler()
X_scaler = scaler.fit_transform(df)
print(X_scaler.shape)
print(X_scaler.mean(axis=0).round(4))
print(X_scaler.std(axis=0).round(4))

km = KMeans(n_clusters=3, random_state=42, n_init=10)
km.fit(X_scaler)
print(np.round(km.inertia_, 2))
print(km.labels_[:10])
print(pd.Series(km.labels_).value_counts().sort_index().to_dict())

print(km.cluster_centers_.shape)
print(km.cluster_centers_.round(2))

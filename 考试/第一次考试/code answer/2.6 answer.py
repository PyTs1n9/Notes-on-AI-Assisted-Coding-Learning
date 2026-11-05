import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()          # 30 个特征 + target（0 = 恶性，1 = 良性）
print(df.shape)                 # (569, 31)

# 以下为题目提供的脏数据生成代码，不得修改
rng = np.random.RandomState(42)
n = len(df)
df.loc[rng.rand(n) < 0.04, 'mean texture'] = np.nan
df.loc[rng.choice(n, 8, replace=False), 'mean radius'] = -1.0
df.loc[rng.choice(n, 6, replace=False), 'mean perimeter'] *= 100
df = pd.concat([df, df.sample(12, random_state=0)], ignore_index=True)

import matplotlib.pyplot as plt

print(df['target'].value_counts().sort_index())     # 0: 212, 1: 357
print(df.groupby('target')[['mean radius', 'mean texture', 'mean perimeter']]
        .agg(['mean', 'std']).round(2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for label, name, color in [(0, 'malignant (0)', 'red'), (1, 'benign (1)', 'blue')]:
    axes[0].hist(df.loc[df['target'] == label, 'mean radius'],
                 bins=25, alpha=0.6, label=name, color=color)
axes[0].set_title('mean radius by class')
axes[0].set_xlabel('mean radius')
axes[0].set_ylabel('count')
axes[0].legend()

axes[1].scatter(df['mean radius'], df['mean texture'],
                c=df['target'], cmap='coolwarm', s=12, alpha=0.7)
axes[1].set_title('mean radius vs mean texture')
axes[1].set_xlabel('mean radius')
axes[1].set_ylabel('mean texture')
plt.tight_layout()
plt.show()

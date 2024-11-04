import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 데이터 불러오기
iris = datasets.load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)

# 데이터 표준화
scaler = StandardScaler()
df.loc[:, :] = scaler.fit_transform(df)

# KMeans 군집화
kmeans = KMeans(n_clusters=3, random_state=7)
kmeans.fit(df)

# 군집 결과를 DataFrame에 추가
df['cluster'] = kmeans.labels_

# PCA로 2D로 축소
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df.drop('cluster', axis=1))

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['cluster'], cmap='viridis', marker='o', edgecolor='k')
plt.title('KMeans Clustering with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

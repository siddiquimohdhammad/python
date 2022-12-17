from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("darkgrid")

X, y = make_blobs(n_samples=500, centers=5, cluster_std = 1.00)
sns.scatterplot(x=X[:,0], y=X[:,1], c =["green"])
# plt.show()

model = KMeans(n_clusters=5)
model.fit(X)

plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=100, c='black')
plt.show()

"""
Hello world K means clustering
~~~~~~~~~~~
"""
print(__doc__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

df = pd.read_csv ('data_1024.mod.002.csv')
#print df

f1 = df['Distance_Feature'].values
f2 = df['Speeding_Feature'].values

n_clusters = 3

X = np.matrix(zip(f1, f2))
k_means = KMeans(n_clusters=n_clusters).fit(X)

k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)


plt.figure(1, figsize=(4, 3))
colors = ['#4EACC5', '#FF9C34', '#4E9A06']
# KMeans plot 
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
             markerfacecolor=col, marker='.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
#plt.set_title('KMeans')
#plt.set_xticks(())
#plt.set_yticks(())

plt.show()




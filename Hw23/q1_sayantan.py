from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random as rand
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster import cluster_visualizer_multidim
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

class_A = []
class_B = []
 
for i in range(100):
	class_A.append([float(rand.randint(1,101)),float(rand.randint(1,101))])
	class_B.append([float(rand.randint(1,101)),float(rand.randint(1,101))])

inp = np.vstack([class_A,class_B])
avg = []
arr_k = []
# Prepare initial centers using K-Means++ method.
for h in range(5):
	k = rand.randint(2,200)
	initial_centers = kmeans_plusplus_initializer(inp, k).initialize()

	# Create instance of K-Means algorithm with prepared centers.
	kmeans_instance = kmeans(inp, initial_centers)

	# Run cluster analysis and obtain results.
	kmeans_instance.process()
	clusters = kmeans_instance.get_clusters()

	cluster_labels = [0 for i in range(200)]

	for i in range(len(clusters)):
		for j in range(len(clusters[i])) :
			cluster_labels[clusters[i][j]] = i

	silhouette_avg = silhouette_score(inp, cluster_labels)
	print('silhouette average for k =',k,':',silhouette_avg)
	avg.append(silhouette_avg)
	arr_k.append(k)

	final_centers = kmeans_instance.get_centers()

	# use this for 1D,2D or 3D clustering
	# Visualize obtained results
	kmeans_visualizer.show_clusters(inp, clusters, final_centers)

plt.bar(arr_k, avg, width = 0.8, color = ['red', 'green'])
plt.show()
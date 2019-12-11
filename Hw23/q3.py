from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random as rand
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster import cluster_visualizer_multidim
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_points = len(lines)
    dim_points = 28 * 28
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)
    
    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [ int(x) for x in num[1:] ]
        
    return (data, labels)

train_data, train_labels = read_data("sample_train.csv")
test_data, test_labels = read_data("sample_test.csv")
print(train_data.shape, test_data.shape)
print(train_labels.shape, test_labels.shape)

N = train_data.shape[0]
M = train_data.shape[1]
X = 10
S = 100
arr_cnt = np.zeros(X)
arr_data = np.zeros((X,S,M))
arr_labels = np.zeros((X,S))

tN = test_data.shape[0]
tM = test_data.shape[1]
art_cnt = np.zeros(X)
art_data = np.zeros((X,int(tN/X),tM))
art_labels = np.zeros((X,int(tN/X)))

for i in range(N):
    v = int(train_labels[i])
    if arr_cnt[v] == S :
        continue
    arr_data[v,int(arr_cnt[v]),:] = train_data[i,:]
    arr_labels[v,int(arr_cnt[v])] = v
    arr_cnt[v] = arr_cnt[v] + 1

for i in range(tN):
    v = int(test_labels[i])
    art_data[v,int(art_cnt[v]),:] = test_data[i,:]
    art_labels[v,int(art_cnt[v])] = v
    art_cnt[v] = art_cnt[v] + 1

# for i in range(10):
#     print(i,arr_cnt[i])

def get_pca(X,k):
    """
        Get PCA of K dimension using the top eigen vectors 
    """
    pca = PCA(n_components=k)
    X_k = pca.fit_transform(X)
    return X_k

# clustering on full dataset

model_train_data = arr_data[0]
model_train_label = arr_labels[0]

for i in range(1,10,1):
    model_train_data = np.vstack([model_train_data,arr_data[i]])
    model_train_label = np.hstack([model_train_label,arr_labels[i]])

reduced_data = get_pca(model_train_data,2)
inp = []
gt_clusters = [[] for i in range(10)]
col = ['r','g','b','y','k','m','w','c','0.75','#808000']
arr_x = np.zeros(X)
arr_y = np.zeros(X)

for i in range(reduced_data.shape[0]):
    inp.append([float(reduced_data[i,0]),float(reduced_data[i,1])])
    gt_clusters[int(model_train_label[i])].append(i) 
    arr_x[int(model_train_label[i])] += reduced_data[i,0]
    arr_y[int(model_train_label[i])] += reduced_data[i,1]
    plt.scatter(reduced_data[i,0],reduced_data[i,1],c=col[int(model_train_label[i])],marker='.')

gt_centers = []

for i in range(10):
    tmp = []
    tmp.append(arr_x[i]/arr_cnt[i])
    tmp.append(arr_y[i]/arr_cnt[i])
    gt_centers.append(tmp)
plt.title('sample data')
plt.show()

kmeans_visualizer.show_clusters(inp, gt_clusters, gt_centers)

# Visualize clustering results
visualizer = cluster_visualizer_multidim()
visualizer.append_clusters(gt_clusters, inp, marker='o')
# visualizer.append_cluster(noise, inp, marker='x')
# visualizer.set_canvas_title('original clustering : the ground truth')
visualizer.show()

# Prepare initial centers using K-Means++ method.
initial_centers = kmeans_plusplus_initializer(inp, 10).initialize()

# Create instance of K-Means algorithm with prepared centers.
kmeans_instance = kmeans(inp, initial_centers)

# Run cluster analysis and obtain results.
kmeans_instance.process()
clusters = kmeans_instance.get_clusters()
final_centers = kmeans_instance.get_centers()

# use this for 1D,2D or 3D clustering
# Visualize obtained results
kmeans_visualizer.show_clusters(inp, clusters, final_centers)

# Visualize clustering results
visualizer = cluster_visualizer_multidim()
visualizer.append_clusters(clusters, inp, marker='o')
# visualizer.append_cluster(noise, inp, marker='x')
visualizer.show()

# trying on 5 different initializations
avg = []
arr_k = []
for h in range(5):
    k = 10
    initial_centers = kmeans_plusplus_initializer(inp, k).initialize()

    # Create instance of K-Means algorithm with prepared centers.
    kmeans_instance = kmeans(inp, initial_centers)

    # Run cluster analysis and obtain results.
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()

    cluster_labels = [0 for i in range(1000)]

    for i in range(len(clusters)):
        for j in range(len(clusters[i])) :
            cluster_labels[clusters[i][j]] = i

    silhouette_avg = silhouette_score(inp, cluster_labels)
    print('silhouette average for initialization',h,':',silhouette_avg)
    print('adjusted rand score :',adjusted_rand_score(cluster_labels,model_train_label))
    avg.append(silhouette_avg)
    arr_k.append(k)

    final_centers = kmeans_instance.get_centers()

    # use this for 1D,2D or 3D clustering
    # Visualize obtained results
    kmeans_visualizer.show_clusters(inp, clusters, final_centers)

# trying setting the initial centers as the ones of ground truth

kmeans_instance = kmeans(inp, gt_centers)

# Run cluster analysis and obtain results.
kmeans_instance.process()
clusters = kmeans_instance.get_clusters()
cluster_labels = [0 for i in range(1000)]
for i in range(len(clusters)):
    for j in range(len(clusters[i])) :
        cluster_labels[clusters[i][j]] = i
silhouette_avg = silhouette_score(inp, cluster_labels)
print('silhouette average for ground truth initialization :',silhouette_avg)
print('adjusted rand score :',adjusted_rand_score(cluster_labels,model_train_label))
final_centers = kmeans_instance.get_centers()

# use this for 1D,2D or 3D clustering
# Visualize obtained results
kmeans_visualizer.show_clusters(inp, clusters, final_centers)

# Visualize clustering results
visualizer = cluster_visualizer_multidim()
visualizer.append_clusters(clusters, inp, marker='o')
# visualizer.append_cluster(noise, inp, marker='x')
visualizer.show()
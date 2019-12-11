import numpy as np
import matplotlib.pyplot as plt

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

data, labels = read_data("sample_train.csv")

def calc_del(X,C,num): 
    P = X @ C @ C.T - X
    den = np.linalg.norm(P)
    Q = X.T @ P
    E = (Q + Q.T)@C/den
    if num == 1:
        E = (Q + Q.T)@C/den + 0.005 * (X.T @ np.sign(X @ C) + np.sign(C))
    if num == 2:
        E = (Q + Q.T)@C/den + 0.005 * 0.005 * ((X.T @ X @ C)/np.linalg.norm(X @ C) + C/np.linalg.norm(C))
    return E

def grd(data,alpha,iters,num):
    C=np.random.rand(784,2)
    for i in range(iters):
        mask = np.random.choice([False, True], 6000, p=[0.9, 0.1])
        dell = calc_del(data[mask],C,num)
        C = C - alpha*dell
    return C  

# No Regularisation
alpha = 0.000005
iters = 400
descent = grd(data,alpha,iters,0)
projected = data @ descent
reconstruct_gd = projected @ descent.T
for i in range(10):
    d = projected[labels==i]
    plt.scatter(d[:,0],d[:,1],s=2)
plt.show()

# L1 Regularisation
alpha = 0.000005
iters = 400
descent = grd(data,alpha,iters,1)
projected_l1 = data @ descent
reconstruct_l1 = projected @ descent.T
for i in range(10):
    d = projected_l1[labels==i]
    plt.scatter(d[:,0],d[:,1],s=2)
plt.show()

# L2 Regularisation
alpha = 0.000005
iters = 400
descent = grd(data,alpha,iters,2)
projected_l2 = data @ descent
reconstruct_l2 = projected @ descent.T
for i in range(10):
    d = projected_l2[labels==i]
    plt.scatter(d[:,0],d[:,1],s=2)
plt.show()

# Regular PCA
C = np.cov(data.T)
(V, D) = np.linalg.eigh(C)
D = D.T[::-1].T
P = D[:, :2]
projected_pca = data @ P
reconstruct_pca = projected_pca @ P.T
for i in range(10):
    d = projected_pca[labels==i]
    plt.scatter(d[:,0],d[:,1],s=2)
plt.show()

total = 6000
error_gd = np.sum(np.abs(data - reconstruct_gd))/total
error_l1 = np.sum(np.abs(data - reconstruct_l1))/total
error_l2 = np.sum(np.abs(data - reconstruct_l2))/total
error_pca = np.sum(np.abs(data - reconstruct_pca))/total
print("Error with Gradient Descent: ", error_gd)
print("Error with Gradient Descent with L1: ", error_l1)
print("Error with Gradient Descent with L2: ", error_l2)
print("Error with EigenVector PCA: ", error_pca)
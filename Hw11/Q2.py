import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

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
# print(train_data.shape, test_data.shape)
# print(train_labels.shape, test_labels.shape)
data_mean = train_data - np.mean(train_data,axis=0)

def normal_pca(X):
    cov = np.cov(np.transpose(X))
    eigvals,eigvecs = LA.eig(cov)
    inds = eigvals.argsort()[::-1]
    eigvals = eigvals[inds]
    eigvecs = eigvecs[:,inds]
    prine = eigvecs[:,0:2]
    project = np.matmul(X,prine)
    plt.scatter(project[:,0:1],project[:,1:2])
    plt.title("Normal Pca")
    plt.show()

def difffunction(X,w):
    e1 = np.matmul(X,w)
    e1 = np.matmul(e1,np.transpose(w))-X
    e2 = np.matmul(w,w.T)
    e2 = np.matmul(e2,X.T) - X.T
    total = np.matmul(np.transpose(X),e1)@w/LA.norm(e1)
    total -= np.matmul(e2,X)@w/LA.norm(e2)
    return total
def gdpca(X):
    alpha = 1/100000
    w = np.random.rand(X.shape[1],2)
    w,_,_ = LA.svd(w)
    w = w[:,:2]
    iters = 150
    for i in range(iters):
        w = w+alpha*difffunction(X,w)
    project = np.matmul(X,w)
    plt.scatter(-project[:,0:1],-project[:,1:2])
    plt.title("Gradient descent Pca")
    plt.show()

normal_pca(data_mean)
# gdpca(data_mean)




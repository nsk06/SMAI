from numpy import linalg as LA
from math import log,pi
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

points = [[1,1],[-1,-1],[1,-1],[-1,1]]
labels  = [1,1,-1,-1]
points = np.array(points)
labels = np.array(labels)

C = [1e-50,5,1e2]
kernels = ['rbf','poly','linear','sigmoid']
Models = []
for k in kernels:
    temp = []
    for c in C:
        model = SVC(C=c,kernel=k)
        model.fit(points,labels)
        temp.append(model)
    Models.append(temp)

data = []
for i in range(200):
    for j in range(200):
        data.append(np.array([i/100-.99,j/100-.99]).reshape(1,-1))
data = np.array(data)
print(data.shape)
t=0
for i in range(len(Models)):
    m = Models[i]
    for c in range(len(m)):
        t+=1
        class1 = []
        class2 = []
        model = m[c]
        for j in data:
            y=model.predict(j)
            if(y==1):
                class1.append(j)
            else:
                class2.append(j)
        class1 = np.array(class1)
        class2 = np.array(class2)
        plt.subplot(4,3,t)
        plt.scatter(points[0][0],points[0][1],c='r')
        plt.scatter(points[1][0],points[1][1],c='r')
        plt.scatter(points[2][0],points[2][1],c='b')
        plt.scatter(points[3][0],points[3][1],c='b')
        print(class1.shape,class2.shape)
        print(len(class2))
        if(len(class1)!=0):
            plt.scatter(class1[:,0,0],class1[:,0,1],c='r')
        if(len(class2) > 0):
            plt.scatter(class2[:,0,0],class2[:,0,1],c='b')
        plt.title("For kernel " + str(kernels[i]) + "  and c value == " + str(C[c]))
plt.show()

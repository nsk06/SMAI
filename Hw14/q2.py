import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d



points = 500
class1 = np.random.normal(1,1,points)
labels1 = np.ones(points)
class2 = np.random.normal(2,2,points)
labels2 = -np.ones(points)

traindata = np.hstack((class1,class2))
traindata = np.vstack((traindata,np.ones(1000)))
# print(traindata)
trainlabels = np.hstack((labels1,labels2))

x = np.linspace(-20,20,100)
mat1 = np.ones([100,100])
mat2 = np.ones([100,100])

for i in range(100):
    for j in range(100):
        w = np.array([[i-50,j-45]])
        y = np.dot(w,traindata)
        lore = trainlabels-(1/(1+np.exp(-y)))
        mat1[i][j] = np.dot(lore,lore.T)
        lire = trainlabels-y
        mat2[i][j] = np.dot(lire,lire.T)
mp = plt.axes(projection='3d')
mp.contour3D(x,x,mat2,300)
plt.title("Linear Regression")
plt.show()
mp = plt.axes(projection='3d')
mp.contour3D(x,x,mat1,300)
plt.title("Logistic Regression")
plt.show()
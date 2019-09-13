import numpy as np
import matplotlib.pyplot as plt
from random import random
import numpy.linalg as LA
mean1 = [3,3]
mean2 = [7,7]
sigma = [[3,0],[0,3]]
sigma1 = [[3,0],[0,3]]
sigma2 = [[4,0],[0,3]]
x1 = np.linspace(0,10,100)
y1 = 10-x1
classa = np.random.multivariate_normal(mean1,sigma,2000)
classb = np.random.multivariate_normal(mean2,sigma,2000)
print(classa.shape,classb.shape)
classA = [[]]*2
classB = [[]]*2
for i in range(0,classa.shape[0]):
    cur_x = classa[i,0]
    cur_y = classa[i,1]
    if((cur_x>=0 and cur_y>=0 )):
        if((cur_y<=10 and cur_x<=10)):
            classA[0].append(cur_x)
            classA[1].append(cur_y)
    if(len(classA[0])==1000):
        break
for i in range(0,classb.shape[0]):
    cur_x = classb[i,0]
    cur_y = classb[i,1]
    if((cur_x>=0 and cur_y>=0 )):
        if((cur_y<=10 and cur_x<=10)): 
            classB[0].append(cur_x)
            classB[1].append(cur_y)
    if(len(classB[0])==1000):
        break
classA = np.array(classA)
classB = np.array(classB)
print(classA.shape,classB.shape)
classA = classA.T
classB = classB.T
plt.axis('equal')
plt.scatter(classA[:,0],classA[:,1],c='red',marker='o')
plt.scatter(classB[:,0],classB[:,1],c='blue',marker='x')
# plt.axis('equal')
# plt.plot(x1,y1,c='green')
plt.show()

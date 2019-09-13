import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style
import random
def myfunction(mu1,sigma1,mu2,sigma2):
    class1 = np.random.multivariate_normal(mu1,sigma1,1000)
    class2 = np.random.multivariate_normal(mu2,sigma2,1000)

    fig = plt.figure()
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.set_xlabel('x axis')
    ax1.set_ylabel('y axis')
    ax1.set_zlabel('z axis')
    ax1.set_title('3D-Plot')
    ax1.scatter3D(class1[:,0],class1[:,1],class1[:,2],c='r',marker='o')
    ax1.scatter3D(class2[:,0],class2[:,1],class2[:,2],c='g',marker='x')
    ax2 = fig.add_subplot(234)
    ax2.set_title('X-Y Projection')
    ax2.scatter(class1[:,0],class1[:,1],c='r',marker='o')
    ax2.scatter(class2[:,0],class2[:,1],c='g',marker='x')
    ax3 = fig.add_subplot(235)
    ax3.set_title('Y-Z Projection')
    ax3.scatter(class1[:,1],class1[:,2],c='r',marker='o')
    ax3.scatter(class2[:,1],class2[:,2],c='g',marker='x')
    ax4 = fig.add_subplot(236)
    ax4.set_title('X-Z Projection')
    ax4.scatter(class1[:,0],class1[:,2],c='r',marker='o')
    ax4.scatter(class2[:,0],class2[:,2],c='g',marker='x')
    plt.show()

mu1 = np.random.random(3)*6
sigma1 = np.array([[3,0,0],[0,9,0],[0,0,27]])
mu2 = np.random.random(3)*6
myfunction(mu1,sigma1,mu2,sigma1)
mu1 = np.random.random(3)*6
sigma1 = np.array([[3,2,1],[2,9,6],[1,6,27]])
mu2= np.random.random(3)*6
myfunction(mu1,sigma1,mu2,sigma1)
mu1 = np.random.random(3)*6
sigma1 = np.array([[3,0,0],[0,9,0],[0,0,27]])
sigma2 = np.array([[3,2,1],[2,9,6],[1,6,27]])
myfunction(mu1,sigma1,mu1,sigma2)


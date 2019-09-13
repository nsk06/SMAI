import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal
from mpl_toolkits.mplot3d import Axes3D

#Parameters to set
mean1 = [1,4,9]
cov1 = [[1,0,0],[0,4,0],[0,0,9]]

mean2 = [1,2,3]
cov2 = [[1,0,0],[0,4,0],[0,0,9]]

#Create multivariate normal

R1 = np.random.multivariate_normal(mean1,cov1,1000)
X1 = R1[:,0]
Y1 = R1[:,1]
Z1 = R1[:,2]

R2 = np.random.multivariate_normal(mean2,cov2,1000)
X2 = R2[:,0]
Y2 = R2[:,1]
Z2 = R2[:,2]


#Make a 3D plot
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1,projection='3d')
ax1.set_title('3d')
ax1.scatter3D(X1,Y1,Z1,c='red',marker='o')
ax1.scatter3D(X2,Y2,Z2,c='black',marker='x')
ax2 = fig.add_subplot(2,2,2)
ax2.set_title('X-Y')
ax2.scatter(X1,Y1,c='red',marker='o')
ax2.scatter(X2,Y2,c='black',marker='x')
ax3 = fig.add_subplot(2,2,3)
ax3.set_title('Y-Z')
ax3.scatter(Y1,Z1,c='red',marker='o')
ax3.scatter(Y2,Z2,c='black',marker='x')
ax4 = fig.add_subplot(2,2,4)
ax4.set_title('Z-X')
ax4.scatter(Z1,X1,c='red',marker='o')
ax4.scatter(Z2,X2,c='black',marker='x')
plt.show()
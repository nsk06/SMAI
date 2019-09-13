import random
import matplotlib.pyplot as plt
import numpy as np

N = 1000
X = np.linspace(0,10,1000)
Y = [x+1 + (random.random()*2 - 1) for x in X]

mean_x = np.mean(X)
mean_y = np.mean(Y)

X1 = [x-mean_x for x in X]
Y1 = [y-mean_y for y in Y]

a = np.array([X1,Y1])
b = a.transpose()

Conv = a.dot(b)
v,w = np.linalg.eig(Conv)

lam = 10

plt.plot([mean_x,w[0][1]*lam+mean_x],[mean_y,w[1][1]*lam+mean_y])
plt.plot([mean_x,w[0][0]*lam+mean_x],[mean_y,w[1][0]*lam+mean_y])
plt.axis('equal')
plt.scatter(X,Y,c='b')
plt.show()

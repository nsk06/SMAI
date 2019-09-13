import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
x = np.linspace(-10,10,1000)
a,b,c = [2,1,5]
y = (a*x+c)/b
# plt.plot(x,y,'g')
yran = y+np.random.randn(1000)
#print(yran)
a = np.cov(np.array([x,yran]))
#print(a)
u,v = LA.eig(a)
#print(u)
#print(v)
plt.scatter(x,y,c='y')
x1 = np.linspace(-10,10,1000)
a,b = v[0]
y1 = a*x/b
plt.plot(x1,y1,c='r')
x1 = np.linspace(-10,10,1000)
a,b = v[1]
y1 = a*x/b
plt.plot(x1,y1,c='g')
plt.title('2x+y+5=0 in yellow')
plt.show()
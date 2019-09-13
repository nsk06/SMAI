import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


data = []
choice = [-1, 1]

for i in range(100):
	x = np.random.random(2)
	for num in range(len(x)):
		x[num] *= random.choice(choice)
	x = np.append(x,1)	

	data.append(x)

data = np.array(data)
print(data)

x1 = np.array([1,-1,5])

A = 0
B = 0

for i in range(50):
	res = np.dot(x1, data[i])
	if res > 0:
		A += 1
	else:
		B += 1

print(A, B)

A = 0
B = 0

for i in range(50, 100):
	res = np.dot(x1, data[i])
	if res > 0:
		A += 1
	else:
		B += 1

print(A, B)

a, b, c = x1
d = 0

x = np.linspace(-1,1,10)
y = (a*x+c)/b


#fig = plt.figure()
#ax = fig.add_subplot()
plt.scatter(data[:,0], data[:,1])

#ax = fig.gca()
plt.plot(x,y)

# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, Z)
plt.show()




	# print(res)

# print(data)
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

mean1 = [3,3]
mean2 = [7,7]
sigma = [[3,0],[0,3]]
sigma1 = np.array([[3,0],[0,3]])
sigma2 = np.array([[4,0],[0,3]])
xplot = np.linspace(0,10,100)
yplot = 5 - (4*((xplot-3)**2)-3*((xplot-7)**2))/32
classa = np.random.multivariate_normal(mean1,sigma1,3000).T
classA = [[],[]]
for i in range(0,2000):
	if classa[0][i] <= 10 and classa[1][i] <= 10:
		if classa[0][i] >= 0 and classa[1][i] >= 0:
			classA[0].append(classa[0][i])
			classA[1].append(classa[1][i])
	if(len(classA[0]) >= 1000):
		break

classb = np.random.multivariate_normal(mean2,sigma2,3000).T
classB = [[],[]]
for i in range(0,2000):
	if classb[0][i] <= 10 and classb[1][i] <= 10:
		if classb[0][i] >= 0 and classb[1][i] >= 0:
			classB[0].append(classb[0][i])
			classB[1].append(classb[1][i])
	if(len(classB[0]) >= 1000):
		break
fig = plt.figure()
plt.scatter(classA[0],classA[1],color = 'red',marker = 'o')
plt.scatter(classB[0],classB[1],color = 'blue',marker = 'x')
plt.axis('equal')
plt.plot(xplot,yplot,c='green',lw=4)
plt.title('Quadratic boundary')
plt.show()
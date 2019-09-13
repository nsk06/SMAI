import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

xplot = np.linspace(1,11,11)
yplota = ((1.92*xplot + 11.54) + ((1.92*xplot+11.54)**2 + 4*0.67*(1.23*xplot**2+38.43*xplot-205.07))**0.5)/(2*0.67)
yplotb = ((1.92*xplot + 11.54) - ((1.92*xplot+11.54)**2 + 4*0.67*(1.23*xplot**2+38.43*xplot-205.07))**0.5)/(2*0.67)

classA = [[0,0,2,3,3,2,2],[0,1,0,2,3,2,0]]
classB = [[7,8,9,8,7,8,7],[7,6,7,10,10,9,11]]
classA = np.array(classA)
classB = np.array(classB)
print(classA.shape)
plt.scatter(classA[0,:],classA[1,:],color='red',marker='x')
plt.scatter(classB[0,:],classB[1,:],color='blue',marker='o')
plt.plot(xplot,yplota,color='yellow',lw=3)
plt.plot(xplot,yplotb,color='yellow',lw=3)
plt.plot(xplot,yplota-8,color='green',lw=3)
plt.plot(xplot,yplotb+8,color='green',lw=3)
plt.show()
import matplotlib.pyplot as plt
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D


w1 = np.array([[0, 0], [0, 1], [2, 0], [3, 2], [3, 3], [2, 2], [2,0]])
w2 = np.array([[7, 7], [8, 6], [9, 7], [8, 10], [7, 10], [8, 9], [7,11]])

x1 = []
y1 = []

x2 = []
y2 = []

for i in w1:
	x1.append(i[0]) 
	y1.append(i[1])

for i in w2:
	x2.append(i[0]) 
	y2.append(i[1]) 

mu1 = np.mean(w1, axis=0)
mu2 = np.mean(w2, axis=0)

s1 = np.cov(w1.T)
s2 = np.cov(w2.T)

print(s1)
print(s2)

s1_det = np.linalg.det(s1)
s2_det = np.linalg.det(s2)

s1_inv = np.linalg.inv(s1)
s2_inv = np.linalg.inv(s2)

cov_inv_diff = s1_inv - s2_inv
cov_inv_mean = 2*(np.matmul(s1_inv, mu1.T) - np.matmul(s2_inv, mu2.T)).T

mat = cov_inv_diff
vec = cov_inv_mean

c1 = np.matmul(mu1.T, np.matmul(s1_inv, mu1))
c2 = np.matmul(mu2.T, np.matmul(s2_inv, mu2))
c3 = np.log(s2_det/s1_det)
c4 = 100

print(cov_inv_diff)
print(cov_inv_mean)
print(c1)
print(c2)
print(c3)

x = np.linspace(-5.0, 15.0, 100)

yp1 = ((1.92*x + 11.54) + ((1.92*x+11.54)**2 + 4*0.67*(1.23*x**2+38.43*x-205.07))**0.5)/(2*0.67)
yp2 = ((1.92*x + 11.54) - ((1.92*x+11.54)**2 + 4*0.67*(1.23*x**2+38.43*x-205.07))**0.5)/(2*0.67)

yp3 = ((1.92*x + 11.54) + ((1.92*x+11.54)**2 + 4*0.67*(1.23*x**2+38.43*x-205.07 - c4))**0.5)/(2*0.67)
yp4 = ((1.92*x + 11.54) - ((1.92*x+11.54)**2 + 4*0.67*(1.23*x**2+38.43*x-205.07 - c4))**0.5)/(2*0.67)

# print(cov_inv_diff)
# print(cov_inv_mean)

# xlist = np.linspace(-10.0, 15.0, 100)
# ylist = np.linspace(-10.0, 15.0, 100)
# X, Y = np.meshgrid(xlist, ylist)
# Z = mat[0][0] * X * X + (mat[0][1] + mat[1][0]) * X * Y + mat[1][1] * Y * Y + vec[0] * X + vec[1] * Y + c1 - c2 - c3 
# # print(X, Y)

# # Z = X**2 + Y**2
# plt.figure()
# cp = plt.contour(X, Y, Z, levels=[0])
# # plt.colorbar(cp)
plt.scatter(x1, y1, c='r', marker='x')
plt.scatter(x2, y2, c='b', marker='o')
# plt.plot(x, yp1, c='g')
# plt.plot(x, yp2, c='g')

plt.plot(x, yp3)
plt.plot(x, yp4)
plt.show()
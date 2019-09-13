import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import copy

with open('wine.data') as f:
    lines = f.readlines()
data = []
for l in lines:
    temp = []
    l = l.split(',')
    for i in l:
        temp.append(float(i))
    data.append(temp)
data = np.array(data)
classdata = copy.deepcopy(data)
data = data[:,1:]
data = data.T
print(data.shape)
cov = np.cov(data)
# print(cov)
eigvals,eigvecs = LA.eig(cov)
inds = eigvals.argsort()[::-1]
eigvals = eigvals[inds]
eigvecs = eigvecs[:,inds]
# print(eigvals)
plt.plot([i for i in range(1,len(eigvals)+1)],eigvals)
plt.show()
tot = np.sum(eigvals)
temp = 0
ct = 0
for i in eigvals:
    temp+=i
    ct+=1
    if(temp/tot>0.95):
        break
print(ct)
prine = eigvecs[:,0:2]
# print(data.shape,prine.shape)
project = np.matmul(data.T,prine)
# print(classdata[:,0])
c1=c2=c3=0
for i in range(len(classdata[:,0])):
    if(classdata[:,0][i]==1):
        c1=i
    elif(classdata[:,0][i]==2):
        c2=i
    elif(classdata[:,0][i]==3):
        c3=i
# print(c1,c2,c3)
plt.subplot(2,1,1)
plt.scatter(project[0:c1,0],project[0:c1,1],c='r',marker= '$'+'1'+'$')
plt.scatter(project[c1+1:c2,0],project[c1+1:c2,1],c='b',marker='$'+'2'+'$')
plt.scatter(project[c2+1:c3,0],project[c2+1:c3,1],c='g',marker='$'+'3'+'$')
plt.subplot(2,1,2)
plt.scatter(project[0:c1,0],project[0:c1,1],c='r',marker= '$'+'1'+'$')
plt.scatter(project[c1+1:c2,0],project[c1+1:c2,1],c='b',marker='$'+'2'+'$')
plt.scatter(project[c2+1:c3,0],project[c2+1:c3,1],c='g',marker='$'+'3'+'$')
plt.axis('equal')
plt.title("Equal axes")
plt.show()
# print(prine)
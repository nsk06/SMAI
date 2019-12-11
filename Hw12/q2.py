import numpy as np
classa = [[2,2,1],[1,-1,1],[-1,1,1]]
classb = [[1,1,1],[-2,-2,1],[-1,-1,1]]
w = [1,0,-1]
classa = np.array(classa)
classb = np.array(classb)
w = np.array(w)
prev = np.zeros(3)
# print(classa)
j=0
alpha = 0.5
allw = [list(w)]
def error(a,b):
    return np.sqrt(np.mean((a-b)**2))
while(error(w,prev) > 1e-6):
    j+=1
    prev = w
    for i in classa:
        pred = np.dot(prev,i)
        if(pred<0):
            w = w+alpha*(i)
    for i in classb:
        pred = np.dot(prev,i)
        if(pred>=0):
            w = w-alpha*(i)
    if(list(w) in allw):
        print("going in cycles")
        print(j)
        break
    else:
        allw.append(list(w))
    if(j==1000):
        print("Iterations exceed 1000")
        break


import numpy as np
import numpy.linalg as LA
def dataextract(file):
    temp = []
    with open(file) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    for i in lines:
        d = i.split('\t')
        d = np.array(d).astype('float64')
        temp.append(d)
    data = np.array(temp)
    means = np.mean(data,axis = 0)
    standard = []
    for rows in data:
        rd = []
        for i in range(len(rows)):
            rd.append(rows[i]/means[i])
        standard.append(rd)
    return standard

def difffunc(w,x):
    diff = [0]*6
    for i in range(len(x)):
        d = x[i]
        d[5]=1
        d = np.array(d)
        y = x[5]
        diff = np.array(diff)
        diff = (np.dot(w,d)-y)*d
    return diff
def error(a,b):
    return np.sqrt(np.mean((a-b)**2))
def gde(x):
    alpha = 1.7*1e-4
    w = np.zeros(6)
    prev = w
    j=0
    while(error(w,prev)>1.6*1e-4 or np.all(prev == np.zeros(6)) == True):
        j+=1
        # print(j,error(w,prev))
        prev = w
        w = w-alpha*difffunc(w,x)
    print("Iterations = ",j,"and the final w is ",w)

def optimalgde(x):
    w = np.zeros(6)
    prev = w
    j=0
    # print(np.all(prev == np.zeros(6)))
    while(error(w,prev)>1.6*1e-4 or np.all(prev == np.zeros(6)) == True):
        j+=1
        # print(j,error(w,prev))
        prev = w
        hessian = np.zeros([6,6])
        cur_diff = difffunc(w,x)
        for i in range(len(x)):
            d = x[i]
            d[5]=1
            d = np.array(d)
            hessian += np.outer(d,d)
        alpha = (LA.norm(cur_diff)**2/np.dot(cur_diff,np.dot(hessian,cur_diff)))
        print("learning rate",alpha)
        w = w-alpha*cur_diff
    print("Iterations = ",j,"and the final w is ",w)

x = dataextract("airfoil_self_noise.dat")  
print("Optimal GDE")
optimalgde(x)  
print("Normal GDE")
gde(x)






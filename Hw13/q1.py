import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
# points = [[1,1],[2,1],[0,2],[1,2]]
# labels = [1,1,-1,-1]
# points = np.array(points)
# labels = np.array(labels)
# clf =  Perceptron(tol=1e-3, random_state=0)
# a = clf.fit(points,labels,[0,0])
# w = a.get_params()
# w = a.decision_function(points)
# print(w)
# clf = LogisticRegression(tol=1e-3,random_state=0,solver='lbfgs')
# clf.fit(points,labels)
# w = clf.decision_function(points)
# print(w)
def per():
    print("Running Perceptron")
    classb = [[1,1,1],[1,2,1]]
    classa = [[2,20,1],[2,21,1]]
    classa = np.array(classa)
    classb = np.array(classb)
    w = np.array([0,1,-3])
    prev = np.zeros(3)
    # print(classa)
    j=0
    alpha = 0.001
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
        allw.append(list(w))
        if(j==1000):
            print("Iterations exceed 1000")
            break
    # print(w)
    X = []
    Y = []
    for i in range(4):
        X.append(i)
        eq = -(w[0]*i+w[2])/w[1]
        Y.append(eq)
    plt.plot(X,Y,c='r')
def diffforlr(w,x,y,gamma):
    temp = np.zeros(3)
    for i in range(len(x)):
        eq = gamma*np.dot(w,x[i])
        eq = 1/(1+np.exp(-eq))
        temp += (eq-y[i])*x[i]*gamma
    return temp

def Lr():
    print("Logistic Regression")
    x = [[1,1,1],[1,2,1],[2,20,1],[2,21,1]]
    y = [-1,-1,1,1]
    x = np.array(x)
    y = np.array(y)
    w = [0,1,-3]
    for i in range(len(y)):
        if(y[i]<1):
            y[i]= 0
    alpha = .1
    maxiters = 1000
    gamma = 1
    for i in range(maxiters):
        w = w-alpha*diffforlr(w,x,y,gamma)
    X = []
    Y = []
    for i in range(4):
        X.append(i)
        eq = -(w[0]*i+w[2])/w[1]
        Y.append(eq)
    plt.plot(X,Y,c='b')
    plt.plot(x[:,0],x[:,1],'ro')
per()
Lr()
plt.title("Red for perceptron Blue for logistic regression")
plt.show()
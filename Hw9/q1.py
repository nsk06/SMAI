import matplotlib.pyplot as plt
import numpy as np

def diffmae(w,x,y):
    res = np.array([0.0,0.0])
    for i in range(0,1000):
        res += np.sign(np.dot(w,np.array([x[i],1]))-y[i])*np.array([x[i],1])
    return res/1000
def difflogcosh(w,x,y):
    res = np.array([0.0,0.0])
    for i in range(0,1000):
        res += np.tanh(y[i]-np.dot(w,np.array([x[i],1])))*np.array([x[i],1])*-1
    return res/1000
def  diffhuber(w,d,x,y):
    res = np.array([0.0,0.0])    
    for i in range(0,1000):
        if (abs(y[i] - np.dot(w,np.array([x[i],1]))) < d):
            res += (y[i] - np.dot(w,np.array([x[i],1])))* np.array([x[i],1])*-1
        else:
            res += d*np.sign(y[i] - np.dot(w,np.array([x[i],1])))*np.array([x[i],1])*-1
    return res/1000

def lossmae(x,y,w):
    res = np.array([0.0,0.0])    
    for i in range(0,1000):
        res += np.abs(y[i]-np.dot(w,np.array([x[i],1])))
    return res/1000

def losslogcosh(x,y,w):
    res = np.array([0.0,0.0])    
    for i in range(0,1000):
        res += np.log(np.cosh(y[i]-np.dot(w,np.array([x[i],1]))))
    return res/1000

def losshuber(x,y,d,w):
    res = np.array([0.0,0.0])    
    for i in range(0,1000):
        if (abs(y[i] - np.dot(w,np.array([x[i],1]))) < d):
            res += 0.5 * (y[i] - np.dot(w,np.array([x[i],1])))*(y[i] - np.dot(w,np.array([x[i],1])))
        else:
            d * np.abs(y[i] - np.dot(w,np.array([x[i],1]))) - 0.5 * d * d
    return res/1000

x = np.linspace(0,10,1000)
y = x+np.random.normal(0,1,1000)
d = 10
alpha = 0.01
def plotsforl1():
    global x,y,d,alpha
    losses = []
    W = [[],[]]
    w = np.array([0.0,0.0])
    for i in range(0,1000):
        w = w - alpha*diffmae(w,x,y)
        W[0].append(w[0])
        W[1].append(w[1])
        losses.append(lossmae(x,y,w))

    # print(w)
    plt.plot(x,y,c='r',marker='x')
    y_plot = [np.dot(w[0],0)+(w[1]),np.dot(w[0],10)+(w[1])]
    x_plot = [0,10]
    plt.plot(x_plot,y_plot,c='b')
    plt.title('Data for L1 Loss')
    plt.show()
    plt.plot(losses)
    plt.grid(True)
    plt.title('Loss for L1 Loss')
    plt.show()
    plt.plot(W[0],W[1])
    plt.grid(True)
    plt.title('Gradient descent for L1 Loss')
    plt.show()

def plotsforlogcosh():
    global x,y,d,alpha    
    losses = []
    W = [[],[]]
    w = np.array([0.0,0.0])
    for i in range(0,1000):
        w = w - alpha*difflogcosh(w,x,y)
        W[0].append(w[0])
        W[1].append(w[1])
        losses.append(losslogcosh(x,y,w))

    # print(w)
    plt.plot(x,y,c='r',marker='x')
    y_plot = [np.dot(w[0],0)+(w[1]),np.dot(w[0],10)+(w[1])]
    x_plot = [0,10]
    plt.plot(x_plot,y_plot,c='b')
    plt.title('Data for Logcosh Loss')
    plt.show()
    plt.plot(losses)
    plt.grid(True)
    plt.title('Loss for Logcosh Loss')
    plt.show()
    plt.plot(W[0],W[1])
    plt.grid(True)
    plt.title('Gradient descent for Logcosh Loss')
    plt.show()

def plotsforhuber():
    global x,y,d,alpha    
    losses = []
    W = [[],[]]
    w = np.array([0.0,0.0])
    for i in range(0,1000):
        w = w - alpha*diffhuber(w,d,x,y)
        W[0].append(w[0])
        W[1].append(w[1])
        losses.append(losshuber(x,y,d,w))

    # print(w)
    plt.plot(x,y,c='r',marker='x')
    y_plot = [np.dot(w[0],0)+(w[1]),np.dot(w[0],10)+(w[1])]
    x_plot = [0,10]
    plt.plot(x_plot,y_plot,c='b')
    plt.title('Data for Huber Loss')
    plt.show()
    plt.plot(losses)
    plt.grid(True)
    plt.title('Loss for Huber Loss')
    plt.show()
    plt.plot(W[0],W[1])
    plt.grid(True)
    plt.title('Gradient descent for Huber Loss')
    plt.show()

plotsforl1()
plotsforlogcosh()
plotsforhuber()
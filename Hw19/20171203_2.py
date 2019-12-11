import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from keras.optimizers import RMSprop,Adam
import keras.initializers
from numpy import linalg as LA
import matplotlib.pyplot as plt

def sigmoid(z):
	return 1/(1 + np.exp(-z))
def makedata():
    class1 = np.random.multivariate_normal([0,0], [[1,0],[0,1]], 100)
    class2 = np.random.multivariate_normal([0,0], [[1,0],[0,1]], 100)

    class1_labels = np.ones(100)
    class2_labels = np.zeros(100)

    train_data = np.concatenate((class1[0:80,:],class2[0:80,:]),axis=0)
    train_label = np.concatenate((class1_labels[0:80],class2_labels[0:80]), axis = 0)

    test_data = np.concatenate((class1[80:100,:],class2[80:100,:]),axis=0)
    test_label = np.concatenate((class1_labels[80:100],class2_labels[80:100]))

    return train_data.T,train_label,test_data,test_label
def forwardpass(Weights):  
    global X,Y,alpha,m
    W1 = Weights["W1"]
    W2 = Weights["W2"]
    W3 = Weights["W3"]
    W4 = Weights["W4"]
  
    Z1 = np.dot(W1, X)
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1)
    A2 = sigmoid(Z2)
    Z3 = np.dot(W3, A2)
    A3 = sigmoid(Z3)
    Z4 = np.dot(W4, A3)
    A4 = sigmoid(Z4)
        
    cache = {
        "A1": A1,
        "A2": A2,
        "A3": A3,
        "A4": A4
    }
    return A4, cache
def backpass(Weights,cache,losstype):
    global X,Y
    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]
    A4 = cache["A4"]

    W1 = Weights["W1"]
    W2 = Weights["W2"]
    W3 = Weights["W3"]
    W4 = Weights["W4"]

    dZ4 = 0
    if losstype == 1:
        for i in range(len(Y)):
            if 1 > A4[0][i]*Y[i]:
                dZ4 -= 1*A4*(1-A4)
    else:
        dZ4 = (A4 - Y)*A4*(1-A4)

    dW4 = np.dot(dZ4, A3.T)/m
    db4 = np.sum(dZ4, axis=1, keepdims=True)/m

    dZ3 = np.dot(W4.T, dZ4)*A3*(1-A3)
    dW3 = np.dot(dZ3, A2.T)/m
    db3 = np.sum(dZ3, axis=1, keepdims=True)/m

    dZ2 = np.dot(W3.T, dZ3)*A2*(1-A2)
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m

    dZ1 = np.dot(W2.T, dZ2)*A1*(1-A1)
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m
    
    gradients = {
    "dW1": dW1,
    "dW2": dW2,
    "dW3": dW3,
    "dW4": dW4,
    }
    return gradients
def hingecalc(A):
    global X,Y,alpha,m
    cost = (m - np.dot(A,Y))/m
    return cost

def msecalc(A):
    global X,Y,alpha,m
    cost = np.sum((A-Y)**2)/m
    return cost
def updateWeights(Weights,gradients):
    global X,Y,alpha,m
    W1 = Weights['W1']
    W2 = Weights['W2']
    W3 = Weights['W3']
    W4 = Weights['W4']
    
    W1 = W1 - alpha * gradients["dW1"]
    W2 = W2 - alpha * gradients["dW2"]
    W3 = W3 - alpha * gradients["dW3"]
    W4 = W4 - alpha * gradients["dW4"]
    
    new_Weights = {
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "W4": W4,
    }
    
    return new_Weights
def Model(losstype,epochs):
    global X,Y,inps,hid_1,hid_2,hid_3,outn
    W1 = np.random.randn(hid_1, inps)
    W2 = np.random.randn(hid_2, hid_1)
    W3 = np.random.randn(hid_3,hid_2)
    W4 = np.random.randn(outn, hid_3)

    Weights = {
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "W4": W4,
    }
    utils = []
    for i in range(0,epochs+1):
        out,cache = forwardpass(Weights)
        if(losstype==1):
            cur_util = hingecalc(out)
        else:
            cur_util = msecalc(out)
        utils.append(cur_util)

        gradients = backpass(Weights,cache,losstype)
        Weights = updateWeights(Weights,gradients)

    return Weights,utils

def main():
    global X,Y,hid_1,hid_2,hid_3,out,alpha,epochs
    final,cost = Model(0,epochs)
    final_mse,costmse = Model(0,epochs)
    plt.figure(figsize=(20,10))
    plt.plot(costmse,label = "MSE Loss")
    plt.plot(cost,label = "Hinge Loss")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    X,Y,test_data,test_labels = makedata()
    m = len(X)
    alpha = .05
    epochs = 500
    inps  = 2
    hid_1 = 3
    hid_2 = 5
    hid_3 = 3
    outn = 1
    param = 0.01
    main()
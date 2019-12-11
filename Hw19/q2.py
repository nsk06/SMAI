from numpy import linalg as LA
from math import log,pi
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from sklearn.neural_network import MLPClassifier

def sigmoid(z):
	return 1/(1 + np.exp(-z))

def initialize_parameters(n_x, n_h1,n_h2,n_h3, n_y):
    
    W1 = np.random.randn(n_h1, n_x)
    W2 = np.random.randn(n_h2, n_h1)
    W3 = np.random.randn(n_h3, n_h2)
    W4 = np.random.randn(n_y, n_h3)

    parameters = {
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "W4": W4,
    }

    return parameters

def forward_prop(X, parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    W4 = parameters["W4"]
  
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

def calculate_cost_hinge(A, Y):
    cost = (m - np.dot(A,Y))/m
    return cost

def calculate_cost_MSE(A, y):
    cost = np.sum((A-Y)**2)/m
    return cost

def backward_prop(X, Y, cache, parameters,loss):
    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]
    A4 = cache["A4"]

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    W4 = parameters["W4"]

    dZ4 = 0
    if loss == 1:
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
    
    grads = {
    "dW1": dW1,
    "dW2": dW2,
    "dW3": dW3,
    "dW4": dW4,
    }
    return grads

def update_parameters(parameters, grads, learning_rate):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    
    W1 = W1 - learning_rate * grads["dW1"]
    W2 = W2 - learning_rate * grads["dW2"]
    W3 = W3 - learning_rate * grads["dW3"]
    W4 = W4 - learning_rate * grads["dW4"]
    
    new_parameters = {
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "W4": W4,
    }
    
    return new_parameters

def model(X, Y, n_x, n_h1,n_h2,n_h3, n_y, num_of_iters, learning_rate,loss):
    parameters = initialize_parameters(n_x, n_h1,n_h2,n_h3, n_y)
    
    costs = []
    
    for i in range(0, num_of_iters+1):
        a4, cache = forward_prop(X, parameters)

        if loss == 1:
            cost = calculate_cost_hinge(a4, Y)
        else:
            cost = calculate_cost_MSE(a4, Y)
        costs.append(cost)

        grads = backward_prop(X, Y, cache, parameters,loss)

        parameters = update_parameters(parameters, grads, learning_rate)

        if(i%100 == 0):
            print('Loss after iteration #' + str(i) + ': ' + str(cost))
    return parameters,costs

A_data = np.random.multivariate_normal([0,0], [[1,0],[0,1]], 100)
B_data = np.random.multivariate_normal([0,0], [[1,0],[0,1]], 100)

A_label = np.ones(100)
B_label = np.zeros(100)

train_data = np.concatenate((A_data[0:80,:],B_data[0:80,:]),axis=0)
train_label = np.concatenate((A_label[0:80],B_label[0:80]), axis = 0)

test_data = np.concatenate((A_data[80:100,:],B_data[80:100,:]),axis=0)
test_label = np.concatenate((A_label[80:100],B_label[80:100]))

X = train_data.T
Y = train_label

# No. of training examples
m = X.shape[1]

n_x = 2     #No. of neurons in first layer
n_h1 = 3     #No. of neurons in hidden layer
n_h2 = 5     #No. of neurons in hidden layer
n_h3 = 3     #No. of neurons in hidden layer
n_y = 1     #No. of neurons in output layer
num_of_iters = 300
learning_rate = 0.1

trained_parameters,costs = model(X, Y, n_x, n_h1,n_h2,n_h3, n_y, num_of_iters, learning_rate,0)
trained_parameters,costs_hinge = model(X, Y, n_x, n_h1,n_h2,n_h3, n_y, num_of_iters, learning_rate,1)

plt.figure(figsize=(20,10))
plt.plot(costs,label = "MSE Loss")
plt.plot(costs_hinge,label = "Hinge Loss")
plt.legend()
plt.show()
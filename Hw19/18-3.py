#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_points = len(lines)
    dim_points = 28 * 28
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)
    
    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [ int(x) for x in num[1:] ]
        
    return (data, labels)

train_data, train_labels = read_data("sample_train.csv")
test_data, test_labels = read_data("sample_test.csv")
print(train_data.shape, test_data.shape)
print(train_labels.shape, test_labels.shape)


# In[2]:


data = np.array([train_data[i] for i in range(train_labels.shape[0]) if train_labels[i] == 1])
labels = np.array([0 for i in range(train_labels.shape[0]) if train_labels[i] == 1])
data2 = np.array([train_data[i] for i in range(train_labels.shape[0]) if train_labels[i] == 2])
labels2 = np.array([1 for i in range(train_labels.shape[0]) if train_labels[i] == 2])


# In[3]:


train_data = np.concatenate((data, data2), axis = 0)
train_labels = np.concatenate((labels, labels2), axis = 0)
train_data.shape


# In[4]:


data = np.array([test_data[i] for i in range(test_labels.shape[0]) if test_labels[i] == 1])
labels = np.array([0 for i in range(test_labels.shape[0]) if test_labels[i] == 1])
data2 = np.array([test_data[i] for i in range(test_labels.shape[0]) if test_labels[i] == 2])
labels2 = np.array([1 for i in range(test_labels.shape[0]) if test_labels[i] == 2])
test_data = np.concatenate((data, data2), axis = 0)
test_labels = np.concatenate((labels, labels2), axis = 0)


# In[5]:


import keras
input1 = keras.layers.Input(shape=(784,))
x1 = keras.layers.Dense(1000, activation='tanh')(input1)
x2 = keras.layers.Dense(1000, activation='sigmoid')(x1)
out = keras.layers.Dense(1, activation='sigmoid')(x2)
model = keras.models.Model(inputs=input1, outputs=out)
model.compile(loss='mean_squared_error', optimizer='sgd')


# In[6]:


loss = []
for _ in range(100):
    model.fit(train_data, train_labels)
    o = model.predict(test_data)
    loss.append(np.mean((test_labels-o)*(test_labels-o)))


# In[7]:


plt.plot(np.arange(100), loss)
plt.show()

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


mean1 = [0, 0]
cov1 = [[1, 0], [0, 1]]
mean2 = [0, 1]
cov2 = [[3, 0], [0, 3]]


# In[3]:


class1 = np.random.multivariate_normal(mean1, cov1, 100)
class2 = np.random.multivariate_normal(mean2, cov2, 100)


# In[4]:


data = np.zeros((200, 3))
data[:100, :2] = class1
data[:100, 2] = 0
data[100:, :2] = class2
data[100:, 2] = 1
np.random.shuffle(data)


# In[5]:


train_data = data[:160, :]
test_data = data[160:, :]


# In[6]:


def sig(x):
    return 1/(1 + np.exp(-x))


# In[7]:


w111 = 1
w121 = 1
w211 = 1
w221 = 1
w112 = 1
w122 = 1


# In[8]:


def output(x1, x2):
    global w111, w121, w211, w221, w112, w122
    Hin1 = w111*x1 + w121*x2
    Hin2 = w211*x1 + w221*x2
    Hout1 = np.tanh(Hin1)
    Hout2 = np.tanh(Hin2)
    Oin = w112*Hout1 + w122*Hout2
    Oout = sig(Oin)
    return Oout


# In[9]:


def test():
    out = output(test_data[:,0], test_data[:,1])
    return np.mean((test_data[:,2] - out)*(test_data[:,2] - out))


# In[10]:


def train(x, y, itr, l = 1):
    global w111, w121, w211, w221, w112, w122
    Iout1 = x[:,0]
    Iout2 = x[:,1]
    loss = []
    for _ in range(itr):
        Hin1 = w111*Iout1 + w121*Iout2
        Hin2 = w211*Iout1 + w221*Iout2
        Hout1 = np.tanh(Hin1)
        Hout2 = np.tanh(Hin2)
        Oin = w112*Hout1 + w122*Hout2
        Oout = sig(Oin)
        drEO = -2*(y - Oout)
        drOI = sig(Oin)*(1-sig(Oin))

        drHo1Hi1 = 1 - (np.tanh(Hin1)*np.tanh(Hin1))
        drHo2Hi2 = 1 - (np.tanh(Hin2)*np.tanh(Hin2))

        w111 = w111 - l*np.mean(drEO*drOI*w112*drHo1Hi1*Iout1)
        w121 = w121 - l*np.mean(drEO*drOI*w112*drHo1Hi1*Iout2)
        w211 = w211 - l*np.mean(drEO*drOI*w122*drHo2Hi2*Iout1)
        w221 = w221 - l*np.mean(drEO*drOI*w122*drHo2Hi2*Iout2)

        w112 = w112 - l*np.mean(drEO*drOI*Hout1)
        w122 = w122 - l*np.mean(drEO*drOI*Hout2)
        loss.append(test())
        
    plt.plot(np.arange(itr), loss)
    plt.show()


# In[11]:


train(train_data[:,:2], train_data[:,2], 1000)


# In[12]:


w111 = 1
w121 = 1
w211 = 1
w221 = 1
w112 = 1
w122 = 1
b11 = 1
b21 = 1
b12 = 1


# In[13]:


def output_with_bias(x1, x2):
    global w111, w121, w211, w221, w112, w122, b11, b21, b12
    Hin1 = w111*x1 + w121*x2 + b11
    Hin2 = w211*x1 + w221*x2 + b21
    Hout1 = np.tanh(Hin1)
    Hout2 = np.tanh(Hin2)
    Oin = w112*Hout1 + w122*Hout2 + b12
    Oout = sig(Oin)
    return Oout


# In[14]:


def test_with_bias():
    out = output_with_bias(test_data[:,0], test_data[:,1])
    return np.mean((test_data[:,2] - out)*(test_data[:,2] - out))


# In[15]:


def train_with_bias(x, y, itr, l = 1):
    global w111, w121, w211, w221, w112, w122, b11, b21, b12
    Iout1 = x[:,0]
    Iout2 = x[:,1]
    loss = []
    for _ in range(itr):
        Hin1 = w111*Iout1 + w121*Iout2 + b11
        Hin2 = w211*Iout1 + w221*Iout2 + b21
        Hout1 = np.tanh(Hin1)
        Hout2 = np.tanh(Hin2)
        Oin = w112*Hout1 + w122*Hout2 + b12
        Oout = sig(Oin)
        drEO = -2*(y - Oout)
        drOI = sig(Oin)*(1-sig(Oin))

        drHo1Hi1 = 1 - (np.tanh(Hin1)*np.tanh(Hin1))
        drHo2Hi2 = 1 - (np.tanh(Hin2)*np.tanh(Hin2))

        w111 = w111 - l*np.mean(drEO*drOI*w112*drHo1Hi1*Iout1)
        w121 = w121 - l*np.mean(drEO*drOI*w112*drHo1Hi1*Iout2)
        w211 = w211 - l*np.mean(drEO*drOI*w122*drHo2Hi2*Iout1)
        w221 = w221 - l*np.mean(drEO*drOI*w122*drHo2Hi2*Iout2)
        b11 = b11 - l*np.mean(drEO*drOI*w112*drHo1Hi1)
        b21 = b21 - l*np.mean(drEO*drOI*w122*drHo2Hi2)
        
        w112 = w112 - l*np.mean(drEO*drOI*Hout1)
        w122 = w122 - l*np.mean(drEO*drOI*Hout2)
        b12 = b12 - l*np.mean(drEO*drOI)
        loss.append(test_with_bias())
        
    plt.plot(np.arange(itr), loss)
    plt.show()


# In[16]:


train_with_bias(train_data[:,:2], train_data[:,2], 1000)


# In[17]:


import keras


# In[18]:


input1 = keras.layers.Input(shape=(2,))
x1 = keras.layers.Dense(2, activation='tanh')(input1)
out = keras.layers.Dense(1, activation='sigmoid')(x1)


# In[19]:


model = keras.models.Model(inputs=input1, outputs=out)


# In[20]:


model.compile(loss='mean_squared_error', optimizer='sgd')


# In[21]:


loss = []
for _ in range(1000):
    model.fit(train_data[:,:2], train_data[:, 2])
    o = model.predict(test_data[:,:2])
    loss.append(np.mean((test_data[:,2]-o)*(test_data[:,2]-o)))


# In[23]:


plt.plot(np.arange(1000), loss)
plt.show()


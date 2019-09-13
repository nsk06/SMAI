import numpy as np
import matplotlib.pyplot as plt
from random import random

sz = 100
x = np.linspace(1.0, 10.0, num=sz)
u = np.mean(x)
sigma = np.cov(x)

y = []
for i in x:
	y.append(np.sin(i) + 1/(np.sqrt(2*np.pi)*sigma*np.exp(-1/2*((i-u)/sigma)**2)))


k_values = range(2,int(np.floor(sz/2)),10)
error_mean = []
error_variance = []
for k in k_values:
	error = []
	for i in range(0,int(np.floor(sz/k))):
		test_x = x[i*k:(i+1)*k]
		test_y = y[i*k:(i+1)*k]
		train_x = []
		for l in range(0,i*k):
			train_x.append(x[l])
		for l in range((i+1)*k,sz):
			train_x.append(x[l])
		u_test = np.mean(train_x)
		sig_test = np.cov(train_x)
		pred_y = []
		error_itr = []
		for j in range(0,k):
			pred_y.append(np.exp(-1/2*((test_x[j]-u_test)/sig_test)**2)/(np.sqrt(2*np.pi)*sig_test))
			error_itr.append(np.absolute(pred_y[j] - test_y[j]))
		error.append(np.mean(error_itr))
	error_mean.append(np.mean(error))
	error_variance.append(np.cov(error))

plt.plot(k_values,error_mean,c='r')
plt.plot(k_values,error_variance,c='b')
plt.title('N = ' + str(sz))
plt.show()
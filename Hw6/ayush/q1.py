import numpy as np
import matplotlib.pyplot as plt
from random import random
import math

def k_fold(n):
	x = np.linspace(5.0, 15.0, num=n)
	u = np.mean(x)
	sigma = np.cov(x)

	y = []
	for i in x:
		y.append(math.sin(i) + 1/(math.sqrt(2*math.pi)*sigma*math.exp(-1/2*((i-u)/sigma)**2)))


	k_values = range(2,int(math.floor(n/2)),10)
	error_mean = []
	error_variance = []
	for k in k_values:
		error = []
		for i in range(0,int(math.floor(n/k))):
			test_x = x[i*k:(i+1)*k]
			test_y = y[i*k:(i+1)*k]
			train_x = []
			for l in range((i-1)*k,i*k):
				train_x.append(x[l])
			for l in range((i+1)*k,n):
				train_x.append(x[l])
			u_test = np.mean(train_x)
			sig_test = np.cov(train_x)
			pred_y = []
			error_itr = []
			for j in range(0,k):
				pred_y.append(math.exp(-1/2*((test_x[j]-u_test)/sig_test)**2)/(math.sqrt(2*math.pi)*sig_test))
				error_itr.append(np.absolute(pred_y[j] - test_y[j]))
			error.append(np.mean(error_itr))
		error_mean.append(np.mean(error))
		error_variance.append(np.cov(error))

	plt.plot(k_values,error_mean,c='y')
	plt.plot(k_values,error_variance,c='b')
	plt.title('N = ' + str(n))
	plt.show()

k_fold(100)
k_fold(10000)
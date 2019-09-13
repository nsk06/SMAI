import matplotlib.pyplot as plt
import numpy as np


def gradient_descent(x,learning_rate):
	criteria = []
	error = []

	for i in range(200):
		criteria.append(-2*x*learning_rate)
		x = x - learning_rate*2*x
		error.append(abs(x))

	return (criteria,error)



learning_rate = 0.1
x = -2


criteria,error = gradient_descent(x,learning_rate)
criteria1,error1 = gradient_descent(x,1)
criteria2,error2 = gradient_descent(x,1.005)

plt.plot(criteria,label = "Convergence Criteria for learning_rate = 0.1")
plt.plot(error,label = "Error for learning_rate = 0.1")
plt.plot(criteria1,label = "Convergence Criteria for learning_rate = 1")
plt.plot(error1,label = "Error for learning_rate = 1")
plt.plot(criteria2,label = "Convergence Criteria for learning_rate = 1.005, Divergence ")
plt.plot(error2,label = "Error for learning_rate = 1.005")
plt.legend()
plt.show()




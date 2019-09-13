import matplotlib.pyplot as plt
import numpy as np


def gradient_descent(x,alpha):
	criteria = []
	error = []

	max_iterations = 100
	for i in range(100):
		criteria.append(-2*x*alpha)
		x = x - alpha*2*x
		error.append(abs(x-0))

	return (criteria,error)



alpha = 0.1
x = -2


criteria,error = gradient_descent(x,alpha)
criteria1,error1 = gradient_descent(x,1)
criteria2,error2 = gradient_descent(x,1.01)

fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.plot(criteria,label = "Convergence Criteria for alpha = 0.1, given")
ax1.plot(error,label = "Error for alpha = 0.1, given")
ax1.plot(criteria1,label = "Convergence Criteria for alpha = 1, given")
ax1.plot(error1,label = "Error for alpha = 1, given")
ax1.plot(criteria2,label = "Convergence Criteria for alpha = 1.01, given")
ax1.plot(error2,label = "Error for alpha = 1.01, given")
ax1.grid(True)
ax1.legend()

plt.show()



